package com.example.springbootapp.repository;

import com.example.springbootapp.model.User;
import org.springframework.data.repository.CrudRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

/**
 * Repository interface for User entity operations with Redis.
 * 
 * Extends CrudRepository to provide basic CRUD operations
 * and defines additional query methods.
 */
@Repository
public interface UserRepository extends CrudRepository<User, String> {

    /**
     * Find user by email address.
     * Uses @Indexed annotation on User.email field for efficient querying.
     * 
     * @param email the email address to search for
     * @return Optional containing user if found
     */
    Optional<User> findByEmail(String email);

    /**
     * Check if user exists by email address.
     * 
     * @param email the email address to check
     * @return true if user exists with the given email
     */
    boolean existsByEmail(String email);

    /**
     * Find all users and return as List.
     * Overrides the default Iterable return type for convenience.
     * 
     * @return List of all users
     */
    @Override
    List<User> findAll();

    /**
     * Find users by first name.
     * 
     * @param firstName the first name to search for
     * @return List of users with matching first name
     */
    List<User> findByFirstName(String firstName);

    /**
     * Find users by last name.
     * 
     * @param lastName the last name to search for
     * @return List of users with matching last name
     */
    List<User> findByLastName(String lastName);

    /**
     * Find users by age range.
     * 
     * @param minAge minimum age (inclusive)
     * @param maxAge maximum age (inclusive)
     * @return List of users within the age range
     */
    List<User> findByAgeBetween(Integer minAge, Integer maxAge);
}