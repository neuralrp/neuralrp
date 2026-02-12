# Fix: KeyError 'max_context' in Performance Mode Enabled Path

## Problem Analysis

**Error Message:**
```
[ERROR] Exception in performance mode enabled path: 'max_context'
```

**Root Cause:** Line 5382 in `/api/chat` endpoint accesses `CONFIG['max_context']` (top-level key) instead of `CONFIG['context']['max_context']` (nested key).

## Bug Details

### Performance Mode Enabled Path (Line 5382) - INCORRECT
```python
max_context = request.settings.get('max_context', CONFIG['max_context'])
#                                                ^^^^^^^^^^^^^^^^^^^^
#                                                KeyError - this key doesn't exist at top level
```

### Performance Mode Disabled Path (Line 5456) - CORRECT
```python
max_context = request.settings.get('max_context', CONFIG['context']['max_context'])
#                                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#                                                Correct - nested access
```

## Impact

When `resource_manager.performance_mode_enabled` is **True**:
1. Code tries to access `CONFIG['max_context']` 
2. KeyError is raised (top-level key doesn't exist)
3. Exception is caught by our new error handler
4. Returns `{"error": "'max_context'", "_chat_id": "..."}`
5. Frontend receives error response → blank response in UI

## Solution

Change line 5382 from:
```python
max_context = request.settings.get('max_context', CONFIG['max_context'])
```

To:
```python
max_context = request.settings.get('max_context', CONFIG['context']['max_context'])
```

## File to Modify

- `C:\Users\fosbo\neuralrp\main.py`
- Line 5382

## Implementation Steps

1. **Open main.py** in your text editor
2. **Navigate to line 5382**
3. **Find this line:**
   ```python
   max_context = request.settings.get('max_context', CONFIG['max_context'])
   ```
4. **Replace with:**
   ```python
   max_context = request.settings.get('max_context', CONFIG['context']['max_context'])
   ```
5. **Save the file**
6. **Restart NeuralRP server**

## Verification

After applying fix:
- ✅ No more "'max_context'" error in logs
- ✅ Threshold-based summarization works correctly in performance mode
- ✅ AI responses display properly in UI
- ✅ Browser console shows: `[CHAT ID] Updated currentChatId`

## Expected Result

This single-line fix will resolve the blank response issue when performance mode is enabled. The code was already correct in the performance mode disabled path, so only the enabled path needs the correction.
