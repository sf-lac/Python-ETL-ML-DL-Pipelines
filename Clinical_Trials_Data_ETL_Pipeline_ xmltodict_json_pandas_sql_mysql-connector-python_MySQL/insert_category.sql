USE CLINICAL_TRIALS;

SELECT measure_id INTO @measure_id FROM measure ORDER BY measure_id DESC LIMIT 1;

INSERT INTO category(title, grp, val, measure_id) VALUES 
(%(title)s, %(grp)s, %(val)s, %(measure_id)s);

SELECT * FROM category;

