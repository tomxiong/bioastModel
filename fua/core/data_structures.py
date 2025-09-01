"""
Core data structures for FUA
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Callable


@dataclass
class ModelCapabilities:
    """Model capability declaration"""
    input_size_range: Tuple[Tuple[int, int], Tuple[int, int]]  # [(min_h, min_w), (max_h, max_w)]
    recommended_batch_size: Tuple[int, int]  # (min_batch, max_batch)
    supported_optimizers: List[str]
    supported_schedulers: List[str]
    special_preprocessing: List[str]
    memory_requirements: Dict[str, int]  # {'min_memory': 1024, 'recommended_memory': 2048}
    computational_complexity: str  # 'low', 'medium', 'high'
    training_time_estimate: str  # 'fast', 'medium', 'slow'
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelCapabilities':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class ModelMetadata:
    """Model metadata"""
    name: str
    version: str
    architecture_type: str
    parameter_count: int
    computational_complexity: float
    memory_usage: int
    supported_input_sizes: List[Tuple[int, int]]
    performance_metrics: Dict[str, float]
    training_history: List[Any]
    creation_date: datetime
    last_modified: datetime
    author: str
    tags: List[str]
    description: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        data = asdict(self)
        # Convert datetime objects to ISO format strings
        data['creation_date'] = self.creation_date.isoformat()
        data['last_modified'] = self.last_modified.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelMetadata':
        """Create from dictionary"""
        # Convert ISO format strings back to datetime objects
        if isinstance(data['creation_date'], str):
            data['creation_date'] = datetime.fromisoformat(data['creation_date'])
        if isinstance(data['last_modified'], str):
            data['last_modified'] = datetime.fromisoformat(data['last_modified'])
        return cls(**data)


@dataclass
class Error:
    """Error information"""
    type: str
    severity: str  # 'low', 'medium', 'high'
    description: str
    model_name: str
    timestamp: datetime
    metrics: Optional[Dict] = None
    context: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class Improvement:
    """Improvement suggestion"""
    type: str
    description: str
    implementation: Callable
    priority: str  # 'low', 'medium', 'high'
    expected_impact: str  # 'low', 'medium', 'high'
    implementation_complexity: str  # 'easy', 'medium', 'hard'
    estimated_time: str  # 'minutes', 'hours', 'days'
    risk_level: str  # 'low', 'medium', 'high'
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        data = asdict(self)
        # Remove implementation function as it's not serializable
        data.pop('implementation', None)
        return data