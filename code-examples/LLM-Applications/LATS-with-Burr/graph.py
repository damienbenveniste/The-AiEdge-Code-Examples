from burr.core import ApplicationBuilder, expr
from nodes import expand, initialize, get_best_solution


graph = (
    ApplicationBuilder().with_actions(
        initialize=initialize,
        expand=expand,
        get_best_solution=get_best_solution
    ).with_transitions(
        ("initialize", "expand"),
        ("expand", "get_best_solution", expr('root.is_solved or root.height > 5')),
        ("expand", "expand", ~expr('root.is_solved or root.height > 5'))
    )
    .with_entrypoint("initialize")
    .build()
)