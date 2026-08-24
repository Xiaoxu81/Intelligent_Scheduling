from enum import Enum
from typing import Dict, Optional

class ResourceStatus(Enum):
    IDLE = "IDLE"
    BUSY = "BUSY"
    MAINTENANCE = "MAINTENANCE"
    FAULT = "FAULT"

class Resource:
    def __init__(
        self,
        resource_id: str,
        resource_type: str,
        capacity: float = 1.0,
        capabilities: Optional[Dict[str, float]] = None,
        reliability: float = 1.0,
    ):
        self.resource_id = resource_id
        self.resource_type = resource_type
        self.capacity = capacity
        self.capabilities = dict(capabilities or {})
        self.reliability = reliability
        
        self.status = ResourceStatus.IDLE
        self.current_task_id: Optional[str] = None

    def __repr__(self):
        return f"Resource(id={self.resource_id}, type={self.resource_type}, status={self.status.value})"
