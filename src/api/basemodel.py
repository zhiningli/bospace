from pydantic import BaseModel

class SuggestSearchSpaceRequest(BaseModel):
    code_str: str
    top_k: int = 5


class Bounds(BaseModel):
    learning_rate: float
    weight_decay: float
    num_epochs: int
    momentum: float

class BOSearchSpace(BaseModel):
    lower_bound: Bounds
    upper_bound: Bounds
    
class BORequest(BaseModel):
    code_str: str
    search_space: BOSearchSpace
    n_initial_points: int = 5
    n_iter: int = 20
    allow_logging: bool = True
