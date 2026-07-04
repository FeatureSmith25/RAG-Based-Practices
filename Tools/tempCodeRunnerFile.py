ke()
    message.append(tool_message)
    print(message)

llm_with_tool.invoke(message)
print(result.content