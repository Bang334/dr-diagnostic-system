DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'accounts_role_check'
    ) THEN
        ALTER TABLE accounts
            ADD CONSTRAINT accounts_role_check
            CHECK (role IN ('admin', 'doctor', 'patient'));
    END IF;
END $$;

