DROP DATABASE IF EXISTS CLINICAL_TRIALS;

CREATE DATABASE CLINICAL_TRIALS;

USE CLINICAL_TRIALS;

CREATE TABLE clinical_trial (
	clinical_trial_id VARCHAR(20) UNIQUE NOT NULL PRIMARY KEY,
        title VARCHAR(250) NOT NULL,
	description VARCHAR(500) NOT NULL DEFAULT '',
	status VARCHAR(20) NOT NULL,
	start_date VARCHAR(50) NOT NULL,
	completion_date VARCHAR(50), 
	study_type VARCHAR(20),
	study_design_info VARCHAR(250)
);

CREATE TABLE baseline (
	baseline_id INT PRIMARY KEY AUTO_INCREMENT,
	population VARCHAR(20) NOT NULL,
	description VARCHAR(1000) NOT NULL DEFAULT '',
	clinical_trial_id VARCHAR(20)
);

CREATE TABLE measure (
	measure_id INT PRIMARY KEY AUTO_INCREMENT,
	title VARCHAR(50) NOT NULL,
	units VARCHAR(50) NOT NULL,
	param VARCHAR(50) NOT NULL,
	baseline_id INT
);

CREATE TABLE category (
	category_id INT PRIMARY KEY AUTO_INCREMENT,
	title VARCHAR(50) NOT NULL,
	grp VARCHAR(50) NOT NULL,
	val VARCHAR(50) NOT NULL,
	measure_id INT
);

ALTER TABLE baseline
ADD FOREIGN KEY(clinical_trial_id)
REFERENCES clinical_trial(clinical_trial_id)
ON DELETE CASCADE;

ALTER TABLE measure
ADD FOREIGN KEY(baseline_id)
REFERENCES baseline(baseline_id)
ON DELETE CASCADE;

ALTER TABLE category
ADD FOREIGN KEY(measure_id)
REFERENCES measure(measure_id)
ON DELETE CASCADE;
