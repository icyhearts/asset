
import inspect
current_stack = inspect.stack()
stack_info = ""
for idx, frame_info in enumerate(current_stack):
  stack_info += f".{frame_info.function}"
