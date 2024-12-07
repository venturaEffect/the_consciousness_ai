import unreal
from typing import Dict, Any

class VREnvironment:
    def __init__(self):
        self.ue = unreal.EditorLevelLibrary()
        self.world = self.ue.get_editor_world()
        
    def spawn_agent(self, location: Dict[str, float], avatar_type: str):
        # Spawn MetaHuman character
        character = self.ue.spawn_actor_from_class(
            unreal.MetaHumanCharacter,
            unreal.Transform(
                location=unreal.Vector(
                    x=location['x'],
                    y=location['y'],
                    z=location['z']
                )
            )
        )
        return character
        
    def create_interaction_zone(self, radius: float, location: Dict[str, float]):
        # Create interactive area for agent-environment interaction
        trigger = self.ue.spawn_actor_from_class(
            unreal.TriggerVolume,
            unreal.Transform(
                location=unreal.Vector(
                    x=location['x'],
                    y=location['y'],
                    z=location['z']
                )
            )
        )
        return trigger