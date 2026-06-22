from schemas_pwc import Plan

VALID_TOOLS = {"get_fx_rate", "get_key_rate", "get_inflation", "calculate"}

def validate_plan(plan: Plan) -> list[str]:
    errors = []
    for sq in plan.subquestions:
        bad = [t for t in sq.expected_tools if t not in VALID_TOOLS]
        if bad:
            errors.append(
                f"Подвопрос {sq.id}: {', '.join(bad)} не существует "
                f"(доступны: {', '.join(sorted(VALID_TOOLS))})"
            )
    return errors