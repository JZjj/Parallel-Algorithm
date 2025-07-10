package com.example.springbootapp.service;

import com.example.springbootapp.dto.PagedResponseDTO;
import com.example.springbootapp.dto.UserCreateDTO;
import com.example.springbootapp.dto.UserResponseDTO;
import com.example.springbootapp.dto.UserUpdateDTO;

import java.util.List;

/**
 * Service interface for User operations.
 * 
 * Defines the contract for user-related business logic operations.
 */
public interface UserService {

    /**
     * Create a new user.
     * 
     * @param userCreateDTO the user data to create
     * @return UserResponseDTO containing created user information
     * @throws com.example.springbootapp.exception.UserAlreadyExistsException if email already exists
     */
    UserResponseDTO createUser(UserCreateDTO userCreateDTO);

    /**
     * Get user by ID.
     * 
     * @param id the user ID
     * @return UserResponseDTO containing user information
     * @throws com.example.springbootapp.exception.UserNotFoundException if user not found
     */
    UserResponseDTO getUserById(String id);

    /**
     * Get user by email.
     * 
     * @param email the user email
     * @return UserResponseDTO containing user information
     * @throws com.example.springbootapp.exception.UserNotFoundException if user not found
     */
    UserResponseDTO getUserByEmail(String email);

    /**
     * Get all users with pagination.
     * 
     * @param page page number (0-based)
     * @param size number of items per page
     * @return PagedResponseDTO containing paginated user list
     */
    PagedResponseDTO<UserResponseDTO> getAllUsers(int page, int size);

    /**
     * Get all users without pagination.
     * 
     * @return List of all users
     */
    List<UserResponseDTO> getAllUsers();

    /**
     * Update an existing user.
     * 
     * @param id the user ID to update
     * @param userUpdateDTO the user data to update
     * @return UserResponseDTO containing updated user information
     * @throws com.example.springbootapp.exception.UserNotFoundException if user not found
     * @throws com.example.springbootapp.exception.UserAlreadyExistsException if email already exists for another user
     */
    UserResponseDTO updateUser(String id, UserUpdateDTO userUpdateDTO);

    /**
     * Delete a user by ID.
     * 
     * @param id the user ID to delete
     * @throws com.example.springbootapp.exception.UserNotFoundException if user not found
     */
    void deleteUser(String id);

    /**
     * Check if user exists by ID.
     * 
     * @param id the user ID
     * @return true if user exists
     */
    boolean existsById(String id);

    /**
     * Check if user exists by email.
     * 
     * @param email the user email
     * @return true if user exists
     */
    boolean existsByEmail(String email);

    /**
     * Get total count of users.
     * 
     * @return total number of users
     */
    long getUserCount();
}