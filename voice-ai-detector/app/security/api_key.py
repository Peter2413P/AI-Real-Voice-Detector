from fastapi import Header, HTTPException, status

SECRET_API_KEY = "sk_test_123456789"


def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != SECRET_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
