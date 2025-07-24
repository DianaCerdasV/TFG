import numpy as np
from math import pi
import yaml
import yamlloader
import os
import asyncio

# ros libraries
import rclpy
from rclpy.logging import get_logger
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.task import Future


# Interfaces
from rcl_interfaces.msg import ParameterDescriptor
from trajectory_msgs.msg import JointTrajectoryPoint
from oscar_interfaces.srv import ArmControl, GripperControl
from oscar_emdb_interfaces.srv import PerceptionMultiObj as PerceptionSrv
from oscar_emdb_interfaces.msg import PerceptionMultiObj
from gazebo_msgs.srv import GetEntityState, SetEntityState
from geometry_msgs.msg import Pose, Quaternion, Point
from core_interfaces.srv import LoadConfig
from std_msgs.msg import Float32

# Utils
from core.service_client import ServiceClient, ServiceClientAsync
from core.utils import class_from_classname

# Used for the robots's hand position
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import PointStamped

from gazebo_msgs.msg import ModelStates

    
def setup_control_channel(self, simulation):
    """
    Configure the ROS topics/services for command and policy execution.

    :param simulation: Parameters from the config file to setup the control channel.
    :type simulation: dict
    """
    self.ident = simulation["id"]
    topic = simulation["control_topic"]
    classname = simulation["control_msg"]
    message = class_from_classname(classname)
    self.get_logger().info("Subscribing to... " + str(topic))
    self.create_subscription(message, topic, self.new_command_callback, 0)

    # Optional: subscribe to executed policy topic
    topic = simulation.get("executed_policy_topic")
    service_policy = simulation.get("executed_policy_service")
    service_world_reset = simulation.get("world_reset_service")
    if topic:
        self.get_logger().info("Subscribing to... " + str(topic))
        self.create_subscription(message, topic, self.policy_callback, 0)
    if service_policy:
        self.get_logger().info("Creating server... " + str(service_policy))
        classname = simulation["executed_policy_msg"]
        message_policy_srv = class_from_classname(classname)
        self.create_service(
            message_policy_srv,
            service_policy,
            self.policy_service,
            callback_group=self.mdb_commands_cbg,
        )
        self.get_logger().info("Creating perception publisher timer... ")
        # Subscribe to redescribed perceptions
        # Changed the message type to PerceptionMultiObj
        self.sensors_subs = self.create_subscription(
            PerceptionMultiObj,
            "/oscar/redescribed_sensors",
            self.publish_perceptions_callback,
            0
        )

    if service_world_reset:
        classname = simulation["executed_policy_msg"]
        self.message_world_reset = class_from_classname(simulation["world_reset_msg"])
        self.create_service(
            self.message_world_reset,
            service_world_reset,
            self.world_reset_service_callback,
            callback_group=self.mdb_commands_cbg
        )

async def update_perceptions(self):
    """
    Requests the latest perceptions from the sensory system.
    Retries up to 20 times if not successful.
    """
    self.get_logger().info("Updating Perceptions...")
    self.old_perception = self.perception
    valid_perception = False
    trials = 0
    while not valid_perception:
        self.perception = await self.cli_oscar_perception.send_request_async()
        valid_perception = self.perception.success
        trials += 1
        if trials == 20:
            return None
    # Changed the printing to a log message
    self.get_logger().info(
        f"Perception updated: {self.perception.object1.label} at ({self.perception.object1.x_position}, {self.perception.object1.y_position})"
    )

async def publish_perceptions_callback(self, msg: PerceptionMultiObj):
    """
    Publishes the current perceptions in the appropriate topics.

    :param msg: Perception message with the latest perceptions.
    :type msg: cognitive_processes_interfaces.msg.Perception
    """
    self.get_logger().debug("Publishing perceptions")
    # Update perception messages with the latest values
    # This part was added to send the new perceptions
    self.perceptions["object1"].data[0].label = self.perception.object1.label
    self.perceptions["object1"].data[0].x_position = self.perception.object1.x_position
    self.perceptions["object1"].data[0].y_position = self.perception.object1.y_position
    self.perceptions["object1"].data[0].diameter = 0.025
    self.perceptions["object1"].data[0].color = self.perception.object1.color
    self.perceptions["object1"].data[0].state = self.perception.object1.state

    self.perceptions["object2"].data[0].label = self.perception.object2.label
    self.perceptions["object2"].data[0].x_position = self.perception.object2.x_position
    self.perceptions["object2"].data[0].y_position = self.perception.object2.y_position
    self.perceptions["object2"].data[0].diameter = 0.1
    self.perceptions["object2"].data[0].color = self.perception.object2.color
    self.perceptions["object2"].data[0].state = self.perception.object2.state

    self.perceptions["robot_hand"].data[0].state = self.perception.robot_hand.state
    self.perceptions["robot_hand"].data[0].x_position = self.perception.robot_hand.x_position
    self.perceptions["robot_hand"].data[0].y_position = self.perception.robot_hand.y_position

    self.publish_perceptions(self.perceptions)

async def bring_object_near(self):
    """
    Randomly moves the object so that it is within the reach of the robot and not inside the basket.
    """
    basket_x = self.perception.basket.x
    basket_y = self.perception.basket.y

    object_x = np.random.uniform(low=0.2575, high=0.3625) # Changed the limits
    object_y = np.random.uniform(low=-0.458, high=0.0) # Changed the limits

    delta_x = basket_x - object_x
    delta_y = basket_y - object_y
    distance = np.sqrt(delta_x * delta_x + delta_y * delta_y)
    while distance < 0.15:
        object_x = np.random.uniform(low=0.2575, high=0.3625) # Changed the limits
        object_y = np.random.uniform(low=-0.458, high=0.0) # Changed the limits
        delta_x = basket_x - object_x
        delta_y = basket_y - object_y
        distance = np.sqrt(delta_x * delta_x + delta_y * delta_y)

    object_pose = Pose(
        position=Point(x=object_x, y=object_y, z=0.8),
        orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    obj_msg = SetEntityState.Request()
    obj_msg.state.name = "object"
    obj_msg.state.pose = object_pose
    obj_msg.state.reference_frame = "world"

    move_resp = await self.cli_set_state.call_async(obj_msg)
    self.get_logger().debug(f"Moving object. Response: {move_resp.success}")
    self.aprox_object = True  # Set flag indicating the object was approximated

async def random_positions(self):
    """
    Randomly places the basket and the object so that they are not colliding.
    """
    basket_x = np.random.uniform(low=0.25, high=0.35) # Changed the limits
    basket_y = np.random.uniform(low=-0.55, high=0) # Changed the limits

    object_x = np.random.uniform(low=0.1125, high=0.7375) # Changed the limits
    object_y = np.random.uniform(low=-0.7415, high=0) # Changed the limits

    delta_x = basket_x - object_x
    delta_y = basket_y - object_y
    distance = np.sqrt(delta_x * delta_x + delta_y * delta_y)
    while distance < 0.15:
        object_x = np.random.uniform(low=0.1125, high=0.7375) # Changed the limits
        object_y = np.random.uniform(low=-0.7415, high=0) # Changed the limits
        delta_x = basket_x - object_x
        delta_y = basket_y - object_y
        distance = np.sqrt(delta_x * delta_x + delta_y * delta_y)

    basket_pose = Pose(
        position=Point(x=basket_x, y=basket_y, z=0.8),
        orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    object_pose = Pose(
        position=Point(x=object_x, y=object_y, z=0.8),
        orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
    )

    bskt_msg = SetEntityState.Request()
    obj_msg = SetEntityState.Request()

    bskt_msg.state.name = "basket"
    bskt_msg.state.pose = basket_pose
    bskt_msg.state.reference_frame = "world"

    obj_msg.state.name = "object"
    obj_msg.state.pose = object_pose
    obj_msg.state.reference_frame = "world"

    move_resp = await self.cli_set_state.call_async(bskt_msg)
    self.get_logger().debug(move_resp.success)
    move_resp = await self.cli_set_state.call_async(obj_msg)
    self.get_logger().debug(move_resp.success)