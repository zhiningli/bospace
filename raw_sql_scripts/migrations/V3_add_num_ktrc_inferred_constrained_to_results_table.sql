-- Add new enum value 'inferred_constrained' to the existing ENUM type
DO $$ 
BEGIN
    -- Check if the enum value already exists to avoid errors
    IF NOT EXISTS (
        SELECT 1 
        FROM pg_enum 
        WHERE enumtypid = 'result_type_enum'::regtype 
          AND enumlabel = 'ktrc_inferred_constrained'
    ) THEN
        ALTER TYPE result_type_enum ADD VALUE 'ktrc_inferred_constrained';
    END IF;
END $$;
