CHAT_ID_HEADER = "x-chat-id"
CHAT_ID_KEY = "chat_id"
UNKNOWN_SESSION = "UNKNOWN"

# getSessionID tries to retrieve session/chat ID from the body, header or state (as map).
# If none found, returns UNKNOWN string.
def getSessionID(*data):
    funcs = [_getSessionIDFromBody, _getSessionIDFromState, _getSessionIDFromHeader]
    for f in funcs:
        for d in data:
            out = f(d)
            if out != UNKNOWN_SESSION:
                return out
        
    return UNKNOWN_SESSION

def _getSessionIDFromState(data):
    out = data.get(CHAT_ID_KEY, {})
    # Make sure headers is a dict before accessing
    if not isinstance(out, str):
        return UNKNOWN_SESSION
    return out

def _getSessionIDFromBody(data):
    headers = data.get("headers", {})
    # Make sure headers is a dict before accessing
    if not isinstance(headers, dict):
        return UNKNOWN_SESSION
    
    return _getSessionIDFromHeader(headers)

def _getSessionIDFromHeader(data):   
    return data.get("x-chat-id", UNKNOWN_SESSION)

def setSessionIDBody(data, to):     
    data["headers"] = { CHAT_ID_HEADER: to }
def setSessionIDHeader(data, to):   
    data[CHAT_ID_HEADER] = to
def setSessionID(data, to):         
    data[CHAT_ID_KEY] = to
