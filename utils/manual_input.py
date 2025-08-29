import logging

def get_user_input(logger: logging.Logger) -> tuple[str | None, bool]:
    """
    Handle user input with exit condition check.
    
    Args:
        logger: Logger instance for logging input events
    
    Returns:
        tuple: (input_query, should_exit) where input_query is the user's input
               or None if empty, and should_exit is True if user wants to exit
    """
    input_query = input("Input (type 'exit' to quit): ").strip()
    if not input_query:
        logger.info("Empty input, skipping...")
        return None, False
    if input_query.lower() == "exit":
        return None, True
    return input_query, False


