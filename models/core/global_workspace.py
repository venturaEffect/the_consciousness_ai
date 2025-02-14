from typing import Dict, List, Any
from dataclasses import dataclass
import asyncio

@dataclass
class WorkspaceMessage:
    source: str
    content: Any
    priority: float

class GlobalWorkspace:
    def __init__(self):
        self.active_processes: List[str] = []
        self.message_queue = asyncio.Queue()
        self.ignition_count = 0
        
    async def broadcast(self, message: WorkspaceMessage):
        """Broadcast message to all subscribed processes"""
        await self.message_queue.put(message)
        
    async def subscribe(self, process_name: str):
        """Subscribe a process to workspace broadcasts"""
        self.active_processes.append(process_name)

    def broadcast(self, message: str, activation_level: float):
        # Called by modules that want to broadcast events
        if activation_level > 0.8:
            self.ignition_count += 1
        # Actual broadcast logic here

    def get_ignition_count(self) -> int:
        return self.ignition_count