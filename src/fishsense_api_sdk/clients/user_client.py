""" "Client for user-related endpoints of the Fishsense API."""

from typing import List

from fishsense_api_sdk.clients.client_base import ClientBase
from fishsense_api_sdk.models.user import User


class UserClient(ClientBase):
    """Client for interacting with user-related endpoints of the Fishsense API."""

    async def get_by_id(self, user_id: int) -> User | None:
        """Get a user by its ID .

        Args:
            user_id (int): The ID of the user to retrieve.

        Returns:
            User | None: The user retrieved from the API.
        """
        response = await self._get(f"/api/v1/users/{user_id}")
        if response.status_code == 404:
            self.logger.debug("No user found with ID %s", user_id)
            return None
        response.raise_for_status()

        json = response.json()
        if json is None:
            self.logger.debug("No user found with ID %s", user_id)
            return None

        return User.model_validate(json)

    async def get_by_email(self, email: str) -> User | None:
        """Get a user by its email .

        Args:
            email (str): The email of the user to retrieve.

        Returns:
            User | None: The user retrieved from the API.
        """
        response = await self._get(f"/api/v1/users/email/{email}")
        if response.status_code == 404:
            self.logger.debug("No user found with email %s", email)
            return None
        response.raise_for_status()

        json = response.json()
        if json is None:
            self.logger.debug("No user found with email %s", email)
            return None

        return User.model_validate(json)

    async def get_by_label_studio_id(self, label_studio_id: int) -> User | None:
        """Get a user by its Label Studio ID .

        Args:
            label_studio_id (int): The Label Studio ID of the user to retrieve.

        Returns:
            User | None: The user retrieved from the API.
        """
        response = await self._get(f"/api/v1/users/label-studio/{label_studio_id}")
        if response.status_code == 404:
            self.logger.debug(
                "No user found with Label Studio ID %s", label_studio_id
            )
            return None
        response.raise_for_status()

        json = response.json()
        if json is None:
            self.logger.debug(
                "No user found with Label Studio ID %s", label_studio_id
            )
            return None

        return User.model_validate(json)

    async def list_all(self) -> List[User] | None:
        """Get all users .

        Returns:
            List[User] | None: The list of users retrieved from the API.
        """
        response = await self._get("/api/v1/users/")
        response.raise_for_status()

        json = response.json()
        if json is None:
            self.logger.debug("No users found.")
            return None

        return [User.model_validate(user) for user in json]

    async def post(self, user: User) -> int:
        """Create a new user .

        Args:
            user (User): The user object to create.

        Returns:
            int: The ID of the created user.
        """
        response = await self._post(
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

        response = await self._put(
            f"/api/v1/users/{user_id}",
            json=user.model_dump(exclude_unset=True, mode="json"),
        )
        response.raise_for_status()
        return response.json()
