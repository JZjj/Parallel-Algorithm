package com.example.springbootapp.config;

import org.springframework.beans.factory.config.ConfigurableBeanFactory;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Scope;

import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import java.time.LocalDateTime;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Application configuration class for custom beans.
 * 
 * Demonstrates different bean scopes and lifecycle methods.
 */
@Configuration
public class AppConfig {

    /**
     * Singleton scoped bean (default scope).
     * Only one instance will be created per Spring container.
     */
    @Bean
    @Scope(ConfigurableBeanFactory.SCOPE_SINGLETON)
    public ApplicationCounter applicationCounter() {
        return new ApplicationCounter();
    }

    /**
     * Prototype scoped bean.
     * New instance will be created each time this bean is requested.
     */
    @Bean
    @Scope(ConfigurableBeanFactory.SCOPE_PROTOTYPE)
    public RequestCounter requestCounter() {
        return new RequestCounter();
    }

    /**
     * Application startup time bean.
     * Records when the application started.
     */
    @Bean
    public LocalDateTime applicationStartupTime() {
        return LocalDateTime.now();
    }

    /**
     * Sample singleton bean demonstrating lifecycle methods.
     */
    public static class ApplicationCounter {
        private final AtomicLong count = new AtomicLong(0);
        private LocalDateTime createdAt;

        @PostConstruct
        public void init() {
            this.createdAt = LocalDateTime.now();
            System.out.println("ApplicationCounter initialized at: " + createdAt);
        }

        @PreDestroy
        public void destroy() {
            System.out.println("ApplicationCounter destroyed. Final count: " + count.get());
        }

        public long increment() {
            return count.incrementAndGet();
        }

        public long getCount() {
            return count.get();
        }

        public LocalDateTime getCreatedAt() {
            return createdAt;
        }
    }

    /**
     * Sample prototype bean demonstrating per-request instances.
     */
    public static class RequestCounter {
        private final AtomicLong count = new AtomicLong(0);
        private final LocalDateTime createdAt;

        public RequestCounter() {
            this.createdAt = LocalDateTime.now();
        }

        public long increment() {
            return count.incrementAndGet();
        }

        public long getCount() {
            return count.get();
        }

        public LocalDateTime getCreatedAt() {
            return createdAt;
        }
    }
}