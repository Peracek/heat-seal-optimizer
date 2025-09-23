# PostgreSQL Setup for Streamlit Community Cloud

This app now supports persistent data storage using PostgreSQL, which means your sealing attempt data will persist across app restarts on Streamlit Community Cloud.

## Quick Setup Guide

### 1. Choose a PostgreSQL Provider

Free PostgreSQL hosting options:
- **Railway** (railway.app) - 5GB free
- **Neon** (neon.tech) - 10GB free
- **Supabase** (supabase.com) - 500MB free
- **ElephantSQL** (elephantsql.com) - 20MB free

### 2. Create Database

1. Sign up for one of the services above
2. Create a new PostgreSQL database
3. Copy your connection string (usually starts with `postgresql://`)

### 3. Configure Streamlit Secrets

In your Streamlit Community Cloud dashboard:

1. Go to your app settings
2. Click on "Secrets"
3. Add your database connection:

```toml
DATABASE_URL = "postgresql://username:password@hostname:port/database"
```

### 4. Deploy

Your app will automatically:
- Use PostgreSQL when secrets are configured
- Fall back to SQLite for local development
- Create all necessary tables on first run

## Local Development

For local development, the app will continue using SQLite (`user_data.db`). To test with PostgreSQL locally:

1. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`
2. Fill in your PostgreSQL connection details
3. Run `streamlit run app.py`

## Migration Notes

- Existing SQLite data is not automatically migrated
- The app is backward compatible and will work without PostgreSQL
- All existing functionality remains the same
- Data structure is identical between SQLite and PostgreSQL

## Troubleshooting

If you see database errors:
1. Verify your DATABASE_URL is correct
2. Check that your PostgreSQL service is running
3. Ensure your IP is whitelisted (if required by your provider)
4. Check Streamlit logs for specific error messages

The app will automatically fall back to SQLite if PostgreSQL connection fails.