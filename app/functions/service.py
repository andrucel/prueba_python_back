from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ImplicitFunction
from app.functions.schemas import FunctionCreate, FunctionUpdate


async def create_function(session: AsyncSession, owner_id: int, data: FunctionCreate) -> ImplicitFunction:
    function = ImplicitFunction(owner_id=owner_id, **data.model_dump())
    session.add(function)
    await session.commit()
    await session.refresh(function)
    return function


async def list_functions(session: AsyncSession, owner_id: int) -> Sequence[ImplicitFunction]:
    query = (
        select(ImplicitFunction)
        .where(ImplicitFunction.owner_id == owner_id)
        .order_by(ImplicitFunction.id)
    )
    return (await session.scalars(query)).all()


async def get_function(session: AsyncSession, function_id: int, owner_id: int) -> ImplicitFunction | None:
    """Obtiene una función por id, restringida a su propietario."""
    query = select(ImplicitFunction).where(
        ImplicitFunction.id == function_id,
        ImplicitFunction.owner_id == owner_id,
    )
    return await session.scalar(query)


async def update_function(
    session: AsyncSession, function: ImplicitFunction, data: FunctionUpdate
) -> ImplicitFunction:
    changes = data.model_dump(exclude_unset=True, exclude_none=True)
    if not changes:
        return function

    for field, value in changes.items():
        setattr(function, field, value)
    function.version += 1
    await session.commit()
    await session.refresh(function)
    return function
