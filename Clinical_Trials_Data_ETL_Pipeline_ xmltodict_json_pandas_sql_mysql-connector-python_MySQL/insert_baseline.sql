USE CLINICAL_TRIALS;

INSERT INTO baseline(population, description, clinical_trial_id) VALUES 
(%(population)s, %(description)s, %(clinical_trial_id)s);

SELECT * FROM baseline;
