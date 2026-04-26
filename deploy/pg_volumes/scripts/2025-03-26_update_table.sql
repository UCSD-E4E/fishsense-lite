ALTER TABLE canonical_dives
ADD spin_slate BOOL NULL;

GRANT UPDATE (spin_slate) ON canonical_dives TO ccrutchf;