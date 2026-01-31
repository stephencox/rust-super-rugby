//! Data ingestion and storage
//!
//! Scrapers for various data sources and SQLite database management.

pub mod database;
pub mod dataset;
pub mod scrapers;

pub use database::Database;
pub use dataset::RugbyDataset;
