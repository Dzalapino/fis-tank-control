import matplotlib.pyplot as plt
import numpy as np
from typing import Callable

# -----------
# Fuzzy logic
# -----------

OR = 'OR'
AND = 'AND'
NONE = 'NONE'


class FuzzyMembership:
    def __init__(self, name: str, points: list[float]):
        """
        Create a fuzzy membership function with the given points.
        :param name: Name of the membership function
        :param points: List of 3 points that define the triangular membership function
        """
        self.name = name
        self.points = points

    def __call__(self, x):
        """
        Create a triangular membership function with the given points.
        :param x: Value to calculate the membership function for
        :return: Membership function
        """
        if x <= self.points[0]:
            return 1 if self.points[0] == self.points[1] else 0
        elif self.points[0] < x <= self.points[1]:
            return (x - self.points[0]) / (self.points[1] - self.points[0])
        elif self.points[1] < x <= self.points[2]:
            return (self.points[2] - x) / (self.points[2] - self.points[1])
        else:
            return 0

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class FuzzyVariable:
    def __init__(self, name: str, membership_functions: list[FuzzyMembership] = None):
        """
        Create a fuzzy variable with the given membership functions.
        :param name: Name of the fuzzy variable
        :param membership_functions: List of membership functions
        """
        self.name = name
        self.membership_functions = membership_functions if membership_functions is not None else []

    def add_membership_function(self, membership_function):
        """
        Add a membership function to the fuzzy variable.
        :param membership_function: Membership function to add
        """
        if membership_function.name in self.membership_functions:
            raise ValueError(f'Membership function with name "{membership_function.name}" already exists')
        self.membership_functions.append(membership_function)

    def get_membership_function(self, name):
        """
        Get the membership function with the given name.
        :param name: Name of the membership function to get
        :return: Membership function with the given name
        """
        for mf in self.membership_functions:
            if mf.name == name:
                return mf
        raise ValueError(f'Membership function with name "{name}" not found')

    def delete_membership_function(self, name):
        """
        Delete the membership function with the given name.
        :param name: Name of the membership function to delete
        """
        if name not in self.membership_functions:
            raise ValueError(f'Membership function with name "{name}" not found')
        self.membership_functions.remove(self.get_membership_function(name))

    def visualize_membership(self):
        """
        Visualize the membership functions of the fuzzy variable.
        """
        # Define the range of x values
        x = np.linspace(self.membership_functions[0].points[0], self.membership_functions[-1].points[-1], 101)
        # Plot the membership functions
        plt.figure(figsize=(10, 5))
        # Create a plot for each membership function
        for f in self.membership_functions:
            y = [f(i) for i in x]
            plt.plot(x, y, label=f.name)
        plt.title('Membership Functions')
        plt.xlabel('Value')
        plt.ylabel('Membership Degree')
        plt.legend()
        plt.grid(True)
        plt.show()

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class FuzzyRule:
    def __init__(self, antecedent: list[tuple[str, str]], operators: list[str], consequent: Callable[[float, float], float]):
        """
        Create a fuzzy rule with the given antecedent and consequent.
        :param antecedent: Antecedent of the rule
        :param operators: Operators of the rule
        :param consequent: Consequent of the rule which is the function
        """
        self.antecedent = antecedent
        self.operators = operators
        self.consequent = consequent


class FuzzySystem:
    def __init__(self, fuzzy_variables: list[FuzzyVariable] = None, fuzzy_rules: list[FuzzyRule] = None):
        """
        Create a fuzzy system with the given fuzzy variables and rules.
        :param fuzzy_variables: List of fuzzy variables
        :param fuzzy_rules: List of fuzzy rules
        """
        self.fuzzy_variables = fuzzy_variables if fuzzy_variables is not None else []
        self.fuzzy_rules = fuzzy_rules if fuzzy_rules is not None else []

    def add_fuzzy_variable(self, fuzzy_variable):
        """
        Add a fuzzy variable to the fuzzy system.
        :param fuzzy_variable: Fuzzy variable to add
        """
        if fuzzy_variable.name in self.fuzzy_variables:
            raise ValueError(f'Fuzzy variable with name "{fuzzy_variable.name}" already exists')
        self.fuzzy_variables.append(fuzzy_variable)

    def get_fuzzy_variable(self, name):
        """
        Get the fuzzy variable with the given name.
        :param name: Name of the fuzzy variable to get
        :return: Fuzzy variable with the given name
        """
        for fv in self.fuzzy_variables:
            if fv.name == name:
                return fv
        raise ValueError(f'Fuzzy variable with name "{name}" not found')

    def delete_fuzzy_variable(self, name):
        """
        Delete the fuzzy variable with the given name.
        :param name: Name of the fuzzy variable to delete
        """
        if name not in self.fuzzy_variables:
            raise ValueError(f'Fuzzy variable with name "{name}" not found')
        self.fuzzy_variables.remove(self.get_fuzzy_variable(name))

    def infer(self, values: dict[str, float]) -> float:
        """
        Infer the output of the fuzzy system with the given inputs.
        """
        outputs = []
        weights = []
        for rule in self.fuzzy_rules:
            membership_degrees = []
            arguments = []
            for antecedent in rule.antecedent:
                # Get the argument for the consequent function
                arguments.append(values[antecedent[0]])

                # Get the membership function of the fuzzy variable
                mf = self.get_fuzzy_variable(antecedent[0]).get_membership_function(antecedent[1])
                # Calculate the membership degree
                membership_degrees.append(mf(values[antecedent[0]]))

            # print('Adding weight:', weight)
            if len(rule.operators) == 0:
                print('No operator, adding weight of single membership degree:', membership_degrees[0])
                weights.append(membership_degrees[0])
            elif rule.operators[0] == OR:
                print('Operator OR, adding max of membership degrees:', membership_degrees, 'with max:', np.max(membership_degrees))
                weights.append(np.max(membership_degrees))
            else:
                print('Operator AND, adding min of membership degrees:', membership_degrees, 'with min:', np.min(membership_degrees))
                weights.append(np.min(membership_degrees))
            print('Adding output of consequent method with arguments:', arguments, 'and output', rule.consequent(*arguments))
            outputs.append(rule.consequent(*arguments))

        return np.dot(outputs, weights) / sum(weights)


# Define the membership functions
proximity_fv = FuzzyVariable('Proximity')
proximity_fv.add_membership_function(FuzzyMembership('Close', [0, 0, 2.5]))
proximity_fv.add_membership_function(FuzzyMembership('Medium', [0, 2.5, 5]))
proximity_fv.add_membership_function(FuzzyMembership('Far', [2.5, 5, 5]))
# proximity.visualize_membership()

speed_fv = FuzzyVariable('Speed')
speed_fv.add_membership_function(FuzzyMembership('Low', [0, 0, 5]))
speed_fv.add_membership_function(FuzzyMembership('Medium', [0, 5, 10]))
speed_fv.add_membership_function(FuzzyMembership('High', [5, 10, 10]))
# speed.visualize_membership()


# Define the rules output methods
def rule(x: float) -> float:
    return 2*x


# Define the rules
rules = [
    FuzzyRule([(proximity_fv.name, 'Close')], [], rule),
    FuzzyRule([(proximity_fv.name, 'Medium')], [], rule),
    FuzzyRule([(proximity_fv.name, 'Far')], [], rule)
]

# Create the fuzzy system
fs = FuzzySystem([proximity_fv, speed_fv], rules)

# ----------
# Init V-REP
# ----------

import vrep
import sys
import time
from tank import *

vrep.simxFinish(-1)  # Closes all opened connections, in case any prevoius wasnt finished
clientID = vrep.simxStart(
    '127.0.0.1',
    19999,
    True,
    True,
    5000,
    5
)  # Start a connection

if clientID != -1:
    print("Connected to remote API server")
else:
    print("Not connected to remote API server")
    sys.exit("Could not connect")


# Create instance of Tank
tank = Tank(clientID)

proximity_sensors = ["EN", "ES", "NE", "NW", "SE", "SW", "WN", "WS"]
proximity_sensors_handles = [0] * 8

# get handle to proximity sensors
for i in range(len(proximity_sensors)):
    err_code, proximity_sensors_handles[i] = vrep.simxGetObjectHandle(
        clientID,
        "Proximity_sensor_" + proximity_sensors[i],
        vrep.simx_opmode_blocking
    )

# Read and print values from proximity sensors
# First reading should be done with simx_opmode_streaming, further with simx_opmode_buffer parameter
for sensor_name, sensor_handle in zip(proximity_sensors, proximity_sensors_handles):
    err_code, detectionState, detectedPoint, detectedObjectHandle, detectedSurfaceNormalVector = (
        vrep.simxReadProximitySensor(clientID, sensor_handle, vrep.simx_opmode_streaming)
    )

# Initial tank movement
tank.forward(2)

# Main tank loop
while True:
    proximity_left = 0
    proximity_right = 0

    # Read proximity sensor values
    for sensor_name, sensor_handle in zip(proximity_sensors, proximity_sensors_handles):
        err_code, detectionState, detectedPoint, detectedObjectHandle, detectedSurfaceNormalVector = (
            vrep.simxReadProximitySensor(clientID, sensor_handle, vrep.simx_opmode_buffer)
        )
        if err_code == 0:
            proximity = np.linalg.norm(detectedPoint)

            if proximity > 5.0 or proximity < 0.001:  # 5.0 is the maximum detection range of the sensor
                proximity = 5.0

            if sensor_name == "NE":
                proximity_right = proximity
            elif sensor_name == "NW":
                proximity_left = proximity

    if proximity_right == proximity_left == 5.0:
        tank.forward(tank.MaxVel)
    elif proximity_right < 1.5 and proximity_left < 1.5:
        tank.stop()
        tank.turn_left(tank.MaxVel)
    elif proximity_left < proximity_right:
        values = {
            proximity_fv.name: proximity_left
        }
        target_speed = fs.infer(values)
        tank.forward(target_speed)
        tank.turn_right()
    else:
        values = {
            proximity_fv.name: proximity_right
        }
        target_speed = fs.infer(values)
        tank.forward(target_speed)
        tank.turn_left()

