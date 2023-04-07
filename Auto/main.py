def print_assistant_thoughts(assistant_reply):
    global ai_name
    global cfg
    try:
        # Parse and print Assistant response
        assistant_reply_json = fix_and_parse_json(assistant_reply)

        # Check if assistant_reply_json is a string and attempt to parse it into a JSON object
        if isinstance(assistant_reply_json, str):
            try:
                assistant_reply_json = json.loads(assistant_reply_json)

        assistant_thoughts_reasoning = None
        assistant_thoughts_plan = None
        assistant_thoughts_speak = None
        assistant_thoughts_criticism = None
        assistant_thoughts = assistant_reply_json.get("thoughts", {})
        assistant_thoughts_text = assistant_thoughts.get("text")

        if assistant_thoughts:
            assistant_thoughts_reasoning = assistant_thoughts.get("reasoning")
            assistant_thoughts_plan = assistant_thoughts.get("plan")
            assistant_thoughts_criticism = assistant_thoughts.get("criticism")
            assistant_thoughts_speak = assistant_thoughts.get("speak")

        try:
            print(assistant_thoughts_text)
            print(assistant_thoughts_reasoning)
            print(assistant_thoughts_plan)
            print(assistant_thoughts_criticism)
            print(assistant_thoughts_speak)




cfg = Config()
parse_arguments()
ai_name = ""
prompt = construct_prompt()


# print(prompt)
# Initialize variables
full_message_history = []
result = None
next_action_count = 0
# Make a constant:
user_input = "Determine which next command to use, and respond using the format specified above:"

# Initialize memory and make sure it is empty.
# this is particularly important for indexing and referencing pinecone memory
memory = PineconeMemory()
memory.clear()
print('Using memory of type: ' + memory.__class__.__name__)




# Interaction Loop
while True:
    # Send message to AI, get response
    with Spinner("Thinking... "):
        assistant_reply = chat.chat_with_ai(
            prompt,
            user_input,
            full_message_history,
            memory,
            cfg.fast_token_limit) # TODO: This hardcodes the model to use GPT3.5. Make this an argument

    # Print Assistant thoughts
    print_assistant_thoughts(assistant_reply)





    # Get command name and arguments
    try:
        command_name, arguments = cmd.get_command(assistant_reply)
    except Exception as e:
        print_to_console("Error: \n", Fore.RED, str(e))


    memory_to_add = f"Assistant Reply: {assistant_reply} " \
                    f"\nResult: {result} " \
                    f"\nHuman Feedback: {user_input} "

    memory.add(memory_to_add)

    # Check if there's a result from the command append it to the message
    # history
    if result is not None:
        full_message_history.append(chat.create_chat_message("system", result))
        print_to_console("SYSTEM: ", Fore.YELLOW, result)
    else:
        full_message_history.append(
            chat.create_chat_message(
                "system", "Unable to execute command"))
        print_to_console("SYSTEM: ", Fore.YELLOW, "Unable to execute command")

