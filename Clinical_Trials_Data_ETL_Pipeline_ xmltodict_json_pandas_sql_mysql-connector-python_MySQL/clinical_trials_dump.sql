-- MySQL dump 10.13  Distrib 8.0.31, for Linux (x86_64)
--
-- Host: localhost    Database: CLINICAL_TRIALS
-- ------------------------------------------------------
-- Server version	8.0.31-0ubuntu0.22.04.1

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `baseline`
--

DROP TABLE IF EXISTS `baseline`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `baseline` (
  `baseline_id` int NOT NULL AUTO_INCREMENT,
  `population` varchar(20) NOT NULL,
  `description` varchar(1000) NOT NULL DEFAULT '',
  `clinical_trial_id` varchar(20) DEFAULT NULL,
  PRIMARY KEY (`baseline_id`),
  KEY `clinical_trial_id` (`clinical_trial_id`),
  CONSTRAINT `baseline_ibfk_1` FOREIGN KEY (`clinical_trial_id`) REFERENCES `clinical_trial` (`clinical_trial_id`) ON DELETE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `baseline`
--

LOCK TABLES `baseline` WRITE;
/*!40000 ALTER TABLE `baseline` DISABLE KEYS */;
INSERT INTO `baseline` VALUES (1,'age 18 and up','500mg of Fulvestrant will be given IM on days 1, 15, 28, then every 4 weeks as per standard of care (SOC) and 160mg of Enzalutamide will be given, in conjunction with Fulvestrant, PO daily.\r\nFulvestrant with Enzalutamide: 500mg of Fulvestrant will be given IM on days 1, 15, 28, then every 4 weeks as per standard of care (SOC) and 160mg of Enzalutamide will be given PO daily. Patients will receive a tumor biopsy at the start of treatment and 4 weeks after the start of treatment, with an optional 3rd biopsy at the end treatment.','NCT02953860');
/*!40000 ALTER TABLE `baseline` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `category`
--

DROP TABLE IF EXISTS `category`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `category` (
  `category_id` int NOT NULL AUTO_INCREMENT,
  `title` varchar(50) NOT NULL,
  `grp` varchar(50) NOT NULL,
  `val` varchar(50) NOT NULL,
  `measure_id` int DEFAULT NULL,
  PRIMARY KEY (`category_id`),
  KEY `measure_id` (`measure_id`),
  CONSTRAINT `category_ibfk_1` FOREIGN KEY (`measure_id`) REFERENCES `measure` (`measure_id`) ON DELETE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=12 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `category`
--

LOCK TABLES `category` WRITE;
/*!40000 ALTER TABLE `category` DISABLE KEYS */;
INSERT INTO `category` VALUES (1,'Age','B1','61',1),(2,'Female','B1','32',2),(3,'Male','B1','0',2),(4,'American Indian or Alaska Native','B1','0',3),(5,'Asian','B1','1',3),(6,'Native Hawaiian or Other Pacific Islander','B1','0',3),(7,'Black or African American','B1','3',3),(8,'White','B1','27',3),(9,'More than one race','B1','1',3),(10,'Unknown or Not Reported','B1','0',3),(11,'Region of Enrollment','B1','32',4);
/*!40000 ALTER TABLE `category` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `clinical_trial`
--

DROP TABLE IF EXISTS `clinical_trial`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `clinical_trial` (
  `clinical_trial_id` varchar(20) NOT NULL,
  `title` varchar(250) NOT NULL,
  `description` varchar(500) NOT NULL DEFAULT '',
  `status` varchar(20) NOT NULL,
  `start_date` varchar(50) NOT NULL,
  `completion_date` varchar(50) DEFAULT NULL,
  `study_type` varchar(20) DEFAULT NULL,
  `study_design_info` varchar(250) DEFAULT NULL,
  PRIMARY KEY (`clinical_trial_id`),
  UNIQUE KEY `clinical_trial_id` (`clinical_trial_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `clinical_trial`
--

LOCK TABLES `clinical_trial` WRITE;
/*!40000 ALTER TABLE `clinical_trial` DISABLE KEYS */;
INSERT INTO `clinical_trial` VALUES ('NCT02953860','Phase II Trial of Fulvestrant Plus Enzalutamide in ER+/Her2- Advanced Breast Cancer','This is a single arm, non-randomized, open-label phase 2 study designed to evaluate the\r\n      tolerability and clinical activity of adding enzalutamide to fulvestrant treatment in women\r\n      with advanced breast cancer that are ER and/or PR-positive and Her2 normal. In this study 500\r\n      mg of Fulvestrant will be given IM on days 1, 15, 28, then every 4 weeks as per standard of\r\n      care (SOC) and 160mg of Enzalutamide will be, in conjunction with Fulvestrant, PO daily.','Completed','July 6, 2017','April 10, 2020','Interventional','None (Open Label)-Treatment-Single Group Assignment');
/*!40000 ALTER TABLE `clinical_trial` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `measure`
--

DROP TABLE IF EXISTS `measure`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `measure` (
  `measure_id` int NOT NULL AUTO_INCREMENT,
  `title` varchar(50) NOT NULL,
  `units` varchar(50) NOT NULL,
  `param` varchar(50) NOT NULL,
  `baseline_id` int DEFAULT NULL,
  PRIMARY KEY (`measure_id`),
  KEY `baseline_id` (`baseline_id`),
  CONSTRAINT `measure_ibfk_1` FOREIGN KEY (`baseline_id`) REFERENCES `baseline` (`baseline_id`) ON DELETE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=5 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `measure`
--

LOCK TABLES `measure` WRITE;
/*!40000 ALTER TABLE `measure` DISABLE KEYS */;
INSERT INTO `measure` VALUES (1,'Age','years','Median',1),(2,'Sex: Female, Male','Participants','Count of Participants',1),(3,'Race (NIH/OMB)','Participants','Count of Participants',1),(4,'Region of Enrollment','Participants','Count of Participants',1);
/*!40000 ALTER TABLE `measure` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2022-11-22 23:35:11
