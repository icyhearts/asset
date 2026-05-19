import inspect
current_stack = inspect.stack()
stack_info = []
for idx, frame_info in enumerate(current_stack):
    frame = frame_info.frame
    func_name = frame_info.function
    class_name = None
    if "self" in frame.f_locals:
        class_name = type(frame.f_locals["self"]).__name__
    elif "cls" in frame.f_locals:
        class_name = frame.f_locals["cls"].__name__
    else:
        code = frame.f_code
        if hasattr(code, 'co_qualname'):
            qualname = code.co_qualname
            #print(f"qualname:{qualname}")
            if "." in qualname:
                class_name = frame.f_code.co_qualname.rsplit('.', 1)[0]
    if class_name:
        method_name = f"{class_name}::{func_name}"
    else:
        method_name = func_name
    stack_info.append(
        f"{idx},{frame_info.filename}:{frame_info.lineno}:{method_name}")
stack_str = "| ".join(stack_info)
with open('/tmp/stack.txt','w') as ofp:
    ofp.write(stack_str)
