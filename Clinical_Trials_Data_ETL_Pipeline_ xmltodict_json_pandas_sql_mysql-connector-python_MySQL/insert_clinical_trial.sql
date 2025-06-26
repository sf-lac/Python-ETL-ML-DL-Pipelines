USE CLINICAL_TRIALS;

INSERT INTO clinical_trial(clinical_trial_id, title, description, status, start_date, completion_date, study_type, study_design_info) VALUES 
(%(clinical_trial_id)s, %(title)s, %(description)s, %(status)s, %(start_date)s, %(completion_date)s, %(study_type)s, %(study_design_info)s);

SELECT * FROM clinical_trial;
