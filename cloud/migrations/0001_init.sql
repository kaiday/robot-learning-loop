CREATE TABLE IF NOT EXISTS iterations (
  run_id            TEXT    NOT NULL,
  iteration         INTEGER NOT NULL,
  version           INTEGER NOT NULL,
  brain             TEXT,
  heldout_success   REAL,
  candidate_success REAL,
  train_success     REAL,
  accepted          INTEGER NOT NULL,
  changes           TEXT,   -- JSON
  failure_counts    TEXT,   -- JSON
  diagnosis         TEXT,
  params            TEXT,   -- JSON
  seconds           REAL,
  created_at        TEXT    NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (run_id, iteration)
);
