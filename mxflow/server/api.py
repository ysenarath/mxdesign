from datetime import timedelta
from typing import List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.responses import HTMLResponse
from fastapi.security import (
    OAuth2PasswordRequestForm,
    SecurityScopes,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from jose import JWTError, jwt
from pydantic import ValidationError

from mxflow.server.db import db
from mxflow.server.schemas import (
    EnvironmentLoaderInDBSchema,
    ExperimentSchema,
    RunSchema,
    TokenDataSchema,
    TokenSchema,
    UserInDBSchema,
    VariableSchema,
)
from mxflow.server.auth import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    authenticate_user,
    create_access_token,
    oauth2_scheme,
    SECRET_KEY,
    ALGORITHM,
    get_user,
)


app = FastAPI()


async def get_current_user(
    security_scopes: SecurityScopes,
    token: str = Depends(oauth2_scheme),
):
    if security_scopes.scopes:
        authenticate_value = f'Bearer scope="{security_scopes.scope_str}"'
    else:
        authenticate_value = "Bearer"
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": authenticate_value},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_scopes = payload.get("scopes", [])
        token_data = TokenDataSchema(scopes=token_scopes, username=username)
    except (JWTError, ValidationError) as ex:
        raise credentials_exception
    user = get_user(username=token_data.username)
    token_data.scopes.append(user.role)
    if user is None:
        raise credentials_exception
    for scope in security_scopes.scopes:
        if scope not in token_data.scopes:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Not enough permissions",
                headers={"WWW-Authenticate": authenticate_value},
            )
    return user


async def get_current_active_user(
    current_user: UserInDBSchema = Security(
        get_current_user,
        # scopes=["me"],
    )
):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


@app.post("/token", response_model=TokenSchema)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={
            "sub": user.username,
            "scopes": form_data.scopes,
        },
        expires_delta=access_token_expires,
    )
    return {
        "access_token": access_token,
        "token_type": "bearer",
    }


@app.get("/users/me/", response_model=UserInDBSchema)
async def read_users_me(
    current_user: UserInDBSchema = Depends(get_current_active_user),
):
    return current_user


@app.post("/environment")
async def add_environment(
    url: str,
    admin_user: UserInDBSchema = Security(
        get_current_user,
        scopes=["admin"],
    ),
) -> EnvironmentLoaderInDBSchema:
    try:
        env = db.add_environment(url=url)
    except ValueError as ex:
        raise HTTPException(
            status_code=409, detail=f"Found existing env with url {url}"
        )
    return EnvironmentLoaderInDBSchema.from_orm(env)


@app.get("/environments")
async def list_environments(
    page: Optional[int] = 1,
    page_size: Optional[int] = 10,
    current_user: UserInDBSchema = Depends(get_current_active_user),
) -> List[EnvironmentLoaderInDBSchema]:
    pagination = db.list_environments()
    pagination.page_number = page
    pagination.page_size = page_size
    envs = []
    for env in pagination.items:
        env = EnvironmentLoaderInDBSchema.from_orm(env)
        envs.append(env)
    return envs


@app.get("/experiments")
async def list_experiments(
    env_id: int,
    page: Optional[int] = 1,
    page_size: Optional[int] = 10,
    current_user: UserInDBSchema = Depends(get_current_active_user),
) -> List[ExperimentSchema]:
    env = db.get_environment(env_id)
    items = []
    pagination = env.list_experiments()
    pagination.page_number = page
    pagination.page_size = page_size
    for item in pagination:
        try:
            items.append(ExperimentSchema.from_orm(item))
        except ValidationError as ex:
            raise ValueError(str(ex))
    return items


@app.get("/experiments/get")
async def get_experiments_by_name(
    env_id: int,
    name: str,
    current_user: UserInDBSchema = Depends(get_current_active_user),
) -> ExperimentSchema:
    env = db.get_environment(env_id)
    expr = env.get_experiment_by_name(name)
    return ExperimentSchema.from_orm(expr)


@app.get("/experiments/{experiment_id}/runs")
async def list_runs_of_experiment(
    experiment_id: int,
    env_id: int,
    page: Optional[int] = 1,
    page_size: Optional[int] = 10,
    current_user: UserInDBSchema = Depends(get_current_active_user),
) -> List[RunSchema]:
    env = db.get_environment(env_id)
    expr = env.get_experiment(experiment_id)
    runs = []
    pagination = expr.list_runs()
    pagination.page_number = page
    pagination.page_size = page_size
    for run in pagination:
        runs.append(RunSchema.from_orm(run))
    return runs


@app.get("/experiments/{experiment_id}/runs/{run_id}/variables")
async def get_all_variables_of_run(
    env_id: int,
    experiment_id: int,
    run_id: int,
    current_user: UserInDBSchema = Depends(get_current_active_user),
) -> List[VariableSchema]:
    env = db.get_environment(env_id)
    expr = env.get_experiment(experiment_id)
    run = expr.get_run(run_id)
    vars = []
    for var in run.list_variables():
        vars.append(VariableSchema.from_orm(var))
    return vars


app.mount("/static", StaticFiles(directory="mxflow/server/static"), name="static")


templates = Jinja2Templates(directory="mxflow/server/templates")


@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
        },
    )
