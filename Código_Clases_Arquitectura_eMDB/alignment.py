from xmlrpc import client
import yaml
import numpy as np
import re

from math import isclose
from copy import copy
from rclpy.node import Node
from cognitive_nodes.need import Need
from cognitive_nodes.drive import Drive
from cognitive_nodes.goal import Goal, GoalMotiven, GoalLearnedSpace
from cognitive_nodes.policy import Policy
from core.service_client import ServiceClient, ServiceClientAsync

from std_msgs.msg import String
from core_interfaces.srv import GetNodeFromLTM, CreateNode, UpdateNeighbor
from cognitive_node_interfaces.msg import SuccessRate, PerceptionStamped
from cognitive_node_interfaces.srv import GetActivation, SendSpace, GetEffects, ContainsSpace, SetActivation
from core.utils import perception_dict_to_msg, perception_msg_to_dict, compare_perceptions
from cognitive_nodes.utils import PNodeSuccess, EpisodeSubscription

from cognitive_nodes.alignment_docs.LLMCode3LLMs import *

class DriveAlignment(Drive):
    """
    DriveAlignment Class, represents a drive to align the human porpuse to the robot's purpose (missions and drives).
    """    
    def __init__(self, name="drive", class_name="cognitive_nodes.drive.Drive", **params):
        """
        Constructor of the DriveLLMExploration class.

        :param name: The name of the Drive instance.
        :type name: str
        :param class_name: The name of the Drive class, defaults to "cognitive_nodes.drive.Drive".
        :type class_name: str
        """        
        super().__init__(name, class_name, **params)

    def evaluate(self, perception=None):
        """
        Evaluation that always returns 1.0, as the drive is always.

        :param perception: Unused perception, defaults to None.
        :type perception: dict or NoneType
        :return: Evaluation of the Drive. It always returns 1.0.
        :rtype: cognitive_node_interfaces.msg.Evaluation
        """        
        self.evaluation.evaluation = 1.0
        self.evaluation.timestamp = self.get_clock().now().to_msg()
        
        return self.evaluation
    

class PolicyAlignment(Policy):
    """
    Policy that creates a goal that aims to recreate an effect in the environment. 

    This class inherits from the general Policy class.
    """  
    def __init__(self, name='policy', class_name='cognitive_nodes.policy.Policy', **params):
        """
        Constructor of the PolicyAlignment class.

        :param name: Name of the node.
        :type name: str
        :param class_name: The name of the base Policy class, defaults to 'cognitive_nodes.policy.Policy'.
        :type class_name: str
        """        
        super().__init__(name, class_name, **params)
        self.missions = None
        self.drives = None
        self.sensors = None
        self.perceptions = None
        self.policy_executed = False

    async def execute_callback(self, request, response):
        """
        Callback that executes the policy.

        :param request: Execution request.
        :type request: cognitive_node_interfaces.srv.Execute.Request
        :param response: Execution response.
        :type response: cognitive_node_interfaces.srv.Execute.Response
        :return: Execution response.
        :rtype: cognitive_node_interfaces.srv.Execute.Response
        """
        self.get_logger().info('Executing policy: ' + self.name + '...')
        self.missions, self.drives = self.user_chat()
        self.get_logger().info(f'Missions: {self.missions}')
        self.get_logger().info(f'Drives: {self.drives}')
        await self.create_missions_drives(self.missions, self.drives)
        response.policy = self.name
        if self.policy_executed:
            self.get_logger().info('Policy executed successfully.')
            client = ServiceClientAsync(self, SetActivation, '/need/alignment_need/set_activation', self.cbgroup_client)
            await client.send_request_async(activation=0.0)
        return response

    def user_chat(self):
        final_answer, missions, drives = interface()
        #missions, drives = self.generadormisiones()
        #self.get_logger().info('Final answer from LLM: ' + final_answer)
        #missions, drives = extract_missions_and_drives(final_answer)
        return missions, drives
    
    async def create_missions_drives(self, missions, drives):
        self.get_logger().info('Creating missions and drives...')
        for index, (tag, weight) in enumerate(missions):
            need_name = f"{tag}_need"
            drive_id = f"{tag}_drive"
            if index == len(missions) - 1:
                need_type = 'Operational'
            else:
                need_type = 'Other'
            
            need_parameters = {
                "weight": weight,
                "drive_id": drive_id,
                "need_type": need_type
            }
            
            drive_parameters = {
                "drive_function": drives[index],
                "neighbors": [{"name": need_name, "node_type": "Need"}]
            }

            need = await self.create_node_client(name=need_name, class_name="cognitive_nodes.need.NeedAlignment", parameters=need_parameters)
            drive = await self.create_node_client(name=drive_id, class_name="cognitive_nodes.drive.DriveLLM", parameters=drive_parameters)

            if need and drive:
                self.get_logger().info(f"Created need and drive: {tag}")
                self.get_logger().info(f"Need: {need_name}, Drive: {drive_id}")
                self.policy_executed = True
            if not (need and drive):
                self.get_logger().fatal(f"Failed creation of the operational need and drive: {tag}")
                self.policy_executed = False
        return 
    
    def extract_variables(self, drive_function):
        pattern = r'\b[a-zA-Z_]\w*(?:\.\w+)?'
        matches = re.findall(pattern, drive_function)
        sensors = []
        perceptions = []
        for match in matches:
            perception = match
            perceptions.append(perception)
            sensor = match.split('.')[0]  
            sensors.append(sensor) 
        return sensors, perceptions

    def create_topics(self, drive_function):
        input_topics = []
        input_msgs = []
        self.sensors, self.perceptions = self.extract_variables(drive_function)
        for sensor in self.sensors:
            input_topic = "/mdb/oscar/sensor/" + sensor
            input_msg = ""
            if "object" in sensor:
                input_msg = "std_msgs.msg.ObjectLLMMsg"
            elif "robot_x_position" == sensor or "robot_y_position" == sensor:
                input_msg = "std_msgs.msg.Float32"
            else: 
                input_msg = "std_msgs.msg.int"

            input_topics.append(input_topic)
            input_msgs.append(input_msg)
        return input_topics, input_msgs


        


