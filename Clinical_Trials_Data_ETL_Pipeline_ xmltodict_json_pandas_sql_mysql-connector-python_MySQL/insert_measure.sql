USE CLINICAL_TRIALS;

SELECT MAX(baseline_id) INTO @baseline_id FROM baseline;

INSERT INTO measure(title, units, param, baseline_id) VALUES 
(%(title)s, %(units)s, %(param)s, @baseline_id);

SELECT * FROM measure;

