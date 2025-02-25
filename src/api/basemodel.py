from pydantic import BaseModel, Field

class SuggestSearchSpaceRequest(BaseModel):
    code_str: str = Field(..., "code string for analysis")
    top_k: int = Field(5, "number of dataset or model to be considerd as similar")

class BOSearchSpace(BaseModel):
    learning_rate: tuple[float, float] = Field(..., description="Bounds for learning rate.")
    momentum: tuple[float, float]=Field(..., description="Bounds for momentum.")
    weight_decay: tuple[float, float]=Field(..., description="Bounds for weight decay")
    num_epochs: tuple[float, float]=Field(..., description="Bounds for number of epochs")

class BORequest(BaseModel):
    code_str: str
    search_space: BOSearchSpace
    n_initial_points: int = 5
    n_iter: int = 20
    allow_logging: bool = True
