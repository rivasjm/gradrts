from abc import ABC, abstractmethod
from model.analysis_function import Function
from model.system_model import SystemModel


class ParameterHandler(Function):
    """
    Handles the extraction and insertion of optimizable parameters.

    This interface defines the methods for converting between the
    real-time system's internal representation of parameters and
    the list of floats used by the optimization algorithm.
    """
    @abstractmethod
    def extract(self, system: SystemModel) -> list[float]:
        """
        Extracts the optimizable parameters from the system as a list of floats.

        Args:
            system: The real-time system from which to extract parameters.

        Returns:
            A list of floats representing the optimizable parameters.
        """
        pass

    @abstractmethod
    def insert(self, system: SystemModel, x: list[float]):
        """
        Inserts the given parameters (list of floats) into the system.

        Args:
            system: The real-time system into which to insert the parameters.
            x: A list of floats representing the parameters to be inserted.
        """
        pass

class CostFunction(Function):
    """
    Defines the cost function to be minimized by the optimization algorithm.

    Concrete subclasses should implement the `compute` method to
    calculate the cost associated with a given set of parameters
    in the real-time system.
    """
    @abstractmethod
    def compute(self, system: SystemModel, x: list[float]) -> float:
        """
        Computes the cost value for the given parameters in the system.

        Args:
            system: The real-time system being optimized.
            x: A list of floats representing the current parameter values.

        Returns:
            The cost value as a float.
        """
        pass

class StopFunction(Function):
    """
    Defines the criteria for stopping the optimization algorithm.

    Concrete subclasses should implement the `should_stop` method
    to determine if the optimization process should terminate based
    on various factors like the number of iterations, cost improvement,
    or other relevant metrics.
    """
    @abstractmethod
    def should_stop(self, system: SystemModel, x: list[float], cost_value: float, t: int) -> bool:
        """
        Checks if the optimization algorithm should stop.

        Args:
            system: The real-time system being optimized.
            x: A list of floats representing the current parameter values.
            cost_value: The current cost value.
            t: The current iteration number.

        Returns:
            True if the optimization should stop, False otherwise.
        """
        pass

    @abstractmethod
    def solution(self, system: SystemModel):
        """Returns the solution it considers the best"""
        pass

    @abstractmethod
    def solution_cost(self):
        """Returns the cost value of the solution"""
        pass

class GradientFunction(Function):
    """
    Calculates the gradient of the cost function.

    Concrete subclasses should implement the `compute` method to
    determine the gradient of the cost function with respect to
    the provided optimizable parameter values.
    """
    @abstractmethod
    def compute(self, system: SystemModel, x: list[float]) -> list[float]:
        """
        Computes the gradient of the cost function at the given parameters.

        Args:
            system: The real-time system being optimized.
            x: A list of floats representing the current parameter values.

        Returns:
            A list of floats representing the gradient vector.
        """
        pass

class UpdateFunction(Function):
    """
    Defines the update rule for the optimization algorithm.

    Concrete subclasses should implement the `update` method to
    determine how the parameters are updated based on the gradient
    and other relevant information (e.g., iteration number, learning rate).
    """
    @abstractmethod
    def update(self, system: SystemModel, x: list[float], nabla: list[float], t: int) -> list[float]:
        """
        Computes the updated parameter values.

        Args:
            system: The real-time system being optimized.
            x: A list of floats representing the current parameter values.
            nabla: A list of floats representing the gradient vector.
            t: The current iteration number.

        Returns:
            A list of floats representing the updated parameter values.
        """
        pass
