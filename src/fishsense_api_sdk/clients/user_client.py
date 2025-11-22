""" "Client for user-related endpoints of the Fishsense API."""

from typing import List

from fishsense_api_sdk.clients.client_base import ClientBase
from fishsense_api_sdk.models.user import User


class UserClient(ClientBase):
    """Client for interacting with user-related endpoints of the Fishsense API."""

    async def get(
        self, user_id: int | None = None, email: str | None = None
    ) -> List[User] | User | None:
        """Get a user by its ID .

        Args:
            user_id (int): The ID of the user to retrieve.

        Returns:
            User | None: The user retrieved from the API.
        """
        async with self._create_client() as client:
            if user_id is not None:
                response = await client.get(f"/api/v1/users/{user_id}")
                response.raise_for_status()

                json = response.json()
                if json is None:
                    return None

                return User.model_validate(json)

            if email is not None:
                response = await client.get(f"/api/v1/users/email/{email}")
                response.raise_for_status()

                json = response.json()
                if json is None:
                    return None

                return User.model_validate(json)

            response = await client.get("/api/v1/users/")
            response.raise_for_status()

            json = response.json()
            if json is None:
                return None

            return [User.model_validate(user) for user in json]

    async def post(self, user: User) -> int:
        """Create a new user .

        Args:
            user (User): The user object to create.

        Returns:
            int: The ID of the created user.
        """
        async with self._create_client() as client:
            response = await client.post(
                "/api/v1/users/",
                json=user.model_dump(exclude_unset=True, mode="json"),
            )
            response.raise_for_status()
            return response.json()

    async def put(self, user: User) -> int:
        """Create a new user .

        Args:
            user (User): The user object to create.

        Returns:
            int: The ID of the created user.
        """
        user_id = user.id

        async with self._create_client() as client:
            response = await client.put(
                f"/api/v1/users/{user_id}",
                json=user.model_dump(exclude_unset=True, mode="json"),
            )
            response.raise_for_status()
            return response.json()
