-- KABUTO エラー報告テーブル作成とCSV投入
CREATE TABLE
IF NOT EXISTS reports_kabuto
(
  id BIGSERIAL PRIMARY KEY,
  error_code TEXT NOT NULL,
  "timestamp" TIMESTAMPTZ NOT NULL,
  user_message TEXT NOT NULL
);

CREATE INDEX
IF NOT EXISTS idx_reports_kabuto_timestamp ON reports_kabuto
("timestamp");
CREATE INDEX
IF NOT EXISTS idx_reports_kabuto_error_code ON reports_kabuto
(error_code);

-- CSV 取り込み
COPY reports_kabuto
(error_code, "timestamp", user_message)
FROM '/docker-entrypoint-initdb.d/reports_kabuto.csv'
WITH
(FORMAT csv, HEADER true, DELIMITER ',', QUOTE '"', ESCAPE '"');
