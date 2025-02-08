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
        
    async def broadcast(self, message: WorkspaceMessage):
        """Broadcast message to all subscribed processes"""
        await self.message_queue.put(message)
        
    async def subscribe(self, process_name: str):
        """Subscribe a process to workspace broadcasts"""
        self.active_processes.append(process_name)