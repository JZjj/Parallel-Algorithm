package com.example.springbootapp.service;

import com.example.springbootapp.dto.PagedResponseDTO;
import com.example.springbootapp.dto.UserCreateDTO;
import com.example.springbootapp.dto.UserResponseDTO;
import com.example.springbootapp.dto.UserUpdateDTO;
import com.example.springbootapp.exception.UserAlreadyExistsException;
import com.example.springbootapp.exception.UserNotFoundException;
import com.example.springbootapp.model.User;
import com.example.springbootapp.repository.UserRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;

/**
 * Implementation of UserService interface.
 * 
 * Provides business logic for user operations with proper error handling,
 * logging, and transaction management.
 */
@Service
@Transactional
public class UserServiceImpl implements UserService {

    private static final Logger logger = LoggerFactory.getLogger(UserServiceImpl.class);

    private final UserRepository userRepository;

    /**
     * Constructor-based dependency injection.
     * 
     * @param userRepository the user repository
     */
    public UserServiceImpl(UserRepository userRepository) {
        this.userRepository = userRepository;
    }

    @Override
    public UserResponseDTO createUser(UserCreateDTO userCreateDTO) {
        logger.debug("Creating user with email: {}", userCreateDTO.getEmail());

        // Check if user with email already exists
        if (userRepository.existsByEmail(userCreateDTO.getEmail())) {
            throw new UserAlreadyExistsException(userCreateDTO.getEmail());
        }

        // Create new user
        User user = new User(
                userCreateDTO.getEmail(),
                userCreateDTO.getFirstName(),
                userCreateDTO.getLastName(),
                userCreateDTO.getAge()
        );
        user.setId(UUID.randomUUID().toString());

        // Save user
        User savedUser = userRepository.save(user);
        logger.info("User created successfully with id: {}", savedUser.getId());

        return mapToResponseDTO(savedUser);
    }

    @Override
    @Transactional(readOnly = true)
    public UserResponseDTO getUserById(String id) {
        logger.debug("Getting user by id: {}", id);

        User user = userRepository.findById(id)
                .orElseThrow(() -> new UserNotFoundException(id));

        return mapToResponseDTO(user);
    }

    @Override
    @Transactional(readOnly = true)
    public UserResponseDTO getUserByEmail(String email) {
        logger.debug("Getting user by email: {}", email);

        User user = userRepository.findByEmail(email)
                .orElseThrow(() -> new UserNotFoundException("User not found with email: " + email));

        return mapToResponseDTO(user);
    }

    @Override
    @Transactional(readOnly = true)
    public PagedResponseDTO<UserResponseDTO> getAllUsers(int page, int size) {
        logger.debug("Getting all users with pagination - page: {}, size: {}", page, size);

        // Validate pagination parameters
        if (page < 0) {
            throw new IllegalArgumentException("Page number cannot be negative");
        }
        if (size <= 0) {
            throw new IllegalArgumentException("Page size must be positive");
        }

        List<User> allUsers = userRepository.findAll();
        long totalElements = allUsers.size();

        // Calculate pagination
        int startIndex = page * size;
        int endIndex = Math.min(startIndex + size, allUsers.size());

        List<UserResponseDTO> content;
        if (startIndex >= allUsers.size()) {
            content = List.of(); // Empty list for pages beyond available data
        } else {
            content = allUsers.subList(startIndex, endIndex)
                    .stream()
                    .map(this::mapToResponseDTO)
                    .collect(Collectors.toList());
        }

        return new PagedResponseDTO<>(content, page, size, totalElements);
    }

    @Override
    @Transactional(readOnly = true)
    public List<UserResponseDTO> getAllUsers() {
        logger.debug("Getting all users without pagination");

        List<User> users = userRepository.findAll();
        return users.stream()
                .map(this::mapToResponseDTO)
                .collect(Collectors.toList());
    }

    @Override
    public UserResponseDTO updateUser(String id, UserUpdateDTO userUpdateDTO) {
        logger.debug("Updating user with id: {}", id);

        // Check if update DTO has any changes
        if (!userUpdateDTO.hasUpdates()) {
            throw new IllegalArgumentException("No fields provided for update");
        }

        // Get existing user
        User existingUser = userRepository.findById(id)
                .orElseThrow(() -> new UserNotFoundException(id));

        // Check if email is being updated and if it already exists for another user
        if (userUpdateDTO.getEmail() != null && 
            !userUpdateDTO.getEmail().equals(existingUser.getEmail())) {
            
            userRepository.findByEmail(userUpdateDTO.getEmail())
                    .ifPresent(user -> {
                        if (!user.getId().equals(id)) {
                            throw new UserAlreadyExistsException(userUpdateDTO.getEmail());
                        }
                    });
        }

        // Update fields
        if (userUpdateDTO.getEmail() != null) {
            existingUser.setEmail(userUpdateDTO.getEmail());
        }
        if (userUpdateDTO.getFirstName() != null) {
            existingUser.setFirstName(userUpdateDTO.getFirstName());
        }
        if (userUpdateDTO.getLastName() != null) {
            existingUser.setLastName(userUpdateDTO.getLastName());
        }
        if (userUpdateDTO.getAge() != null) {
            existingUser.setAge(userUpdateDTO.getAge());
        }

        existingUser.updateTimestamp();

        // Save updated user
        User updatedUser = userRepository.save(existingUser);
        logger.info("User updated successfully with id: {}", updatedUser.getId());

        return mapToResponseDTO(updatedUser);
    }

    @Override
    public void deleteUser(String id) {
        logger.debug("Deleting user with id: {}", id);

        if (!userRepository.existsById(id)) {
            throw new UserNotFoundException(id);
        }

        userRepository.deleteById(id);
        logger.info("User deleted successfully with id: {}", id);
    }

    @Override
    @Transactional(readOnly = true)
    public boolean existsById(String id) {
        return userRepository.existsById(id);
    }

    @Override
    @Transactional(readOnly = true)
    public boolean existsByEmail(String email) {
        return userRepository.existsByEmail(email);
    }

    @Override
    @Transactional(readOnly = true)
    public long getUserCount() {
        return userRepository.count();
    }

    /**
     * Maps User entity to UserResponseDTO.
     * 
     * @param user the user entity
     * @return UserResponseDTO
     */
    private UserResponseDTO mapToResponseDTO(User user) {
        return new UserResponseDTO(
                user.getId(),
                user.getEmail(),
                user.getFirstName(),
                user.getLastName(),
                user.getAge(),
                user.getCreatedAt(),
                user.getUpdatedAt()
        );
    }
}