# Conversation Report — Templates and Format Verification

This report validates the conversation formatting templates and sequence alignment.

---

## 1. Supported Templates

The pipeline supports two formatting patterns:

### Alpaca (Single-Turn)
```text
### Instruction:
{instruction}

### Input:
{input}

### Response:
{response}
```

### Chat (Multi-Turn)
```text
### System:
{system_prompt}

### User:
{user_query}

### Response:
{assistant_response}
```

---

## 2. Validation Testing

We verified the formatting under various constraints:
1. **Empty Input**: Optional inputs are skipped gracefully with correct delimiters.
2. **Multi-turn Dialogue**: Dialogue sequences are correctly flattened into alternating turns.
3. **Max Turns Truncation**: Setting `max_turns` correctly limits the dialogue context while preserving the system message.
4. **UTF-8 Alignment**: Delimiters are decoded safely with zero boundary misalignment.
