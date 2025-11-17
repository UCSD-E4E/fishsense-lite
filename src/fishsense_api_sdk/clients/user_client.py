""" "Client for user-related endpoints of the Fishsense API."""

from fishsense_api_sdk.clients.client_base import ClientBase
from fishsense_api_sdk.models.user import User


class UserClient(ClientBase):
    """Client for interacting with user-related endpoints of the Fishsense API."""

    def __init__(self, base_url: str, timeout: int):
        super().__init__(base_url, timeout)

    async def get(self, user_id: int) -> User | None:
        """Get a user by its ID .

        Args:
            user_id (int): The ID of the user to retrieve.

        Returns:
            User | None: The user retrieved from the API.
        """
        async with self._create_client() as client:
            response = await client.get(f"/api/v1/users/{user_id}")
            response.raise_for_status()

            json = response.json()
            if json is None:
                return None

            return User.model_validate(json)

    async def put(self, user: User) -> int | None:
        """Create a new user .

        Args:
            user (User): The user object to create.

        Returns:
            int | None: The ID of the created user.
        """
        async with self._create_client() as client:
            response = await client.put(
                "/api/v1/users/{user_id}",
                json=user.model_dump(),
            )
            response.raise_for_status()
            return response.json()
