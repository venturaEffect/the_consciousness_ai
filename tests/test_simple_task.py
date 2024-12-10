# File: /tests/test_simple_tasks.py
"""
Unit tests for Simple Tasks Module

Tests the task completion logic and state evaluation.
"""
import unittest
from simulations.scenarios.simple_tasks import SimpleTask, SimpleTaskManager, reach_waypoint

class TestSimpleTasks(unittest.TestCase):
    def setUp(self):
        self.task_manager = SimpleTaskManager()
        self.task = SimpleTask(
            task_id="task_1",
            description="Reach a waypoint.",
            success_criteria=reach_waypoint
        )
        self.task_manager.add_task(self.task)

    def test_task_completion(self):
        agent_state = {"position": [5, 5], "waypoint": [5, 5]}
        self.task_manager.evaluate_tasks(agent_state)
        self.assertTrue(self.task.completed)

    def test_incomplete_task(self):
        agent_state = {"position": [0, 0], "waypoint": [5, 5]}
        self.task_manager.evaluate_tasks(agent_state)
        self.assertFalse(self.task.completed)

if __name__ == "__main__":
    unittest.main()