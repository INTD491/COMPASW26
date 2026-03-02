# Data report 
This report list out all of the variables, their meaning, attributes,key features within compas-score-two-years dataset.

## All variavles

|Attribute Name         |Data Type  |Description / Meaning                                                         |
|-----------------------|-----------|------------------------------------------------------------------------------|
|age                    |Integer    |Defendant's age at the time of the COMPAS screening.                          |
|age_cat                |Categorical|Age grouped into brackets: Less than 25, 25 - 45, or Greater than 45.         |
|sex                    |Binary     |Biological sex of the defendant (Male or Female).                             |
|race                   |Categorical|Ethnicity (e.g., African-American, Caucasian, Hispanic, Other).               |
|priors_count           |Integer    |Total number of previous criminal convictions or arrests.                     |
|juv_fel_count          |Integer    |Number of prior juvenile felony charges.                                      |
|juv_misd_count         |Integer    |Number of prior juvenile misdemeanor charges.                                 |
|juv_other_count        |Integer    |Number of prior juvenile charges that were neither felony nor misdemeanor.    |
|c_charge_degree        |Categorical|Severity of the current charge (F for Felony, M for Misdemeanor).             |
|c_charge_desc          |String     |Text description of the specific offense (e.g., "Grand Theft").               |
|decile_score           |Integer    |The COMPAS-calculated risk score, ranging from 1 (Low) to 10 (High).          |
|score_text             |Categorical|Interpretation of the decile score: Low, Medium, or High.                     |
|is_recid               |Binary     |Whether the defendant was re-arrested at any point (1 = Yes, 0 = No), this variable also include the two_year_recid .         |
|two_year_recid         |Binary     |Primary Target: Did the defendant re-offend within 2 years? (1 = Yes, 0 = No).|
|days_b_screening_arrest|Integer    |Days between the arrest and the COMPAS screening (Used for data cleaning).    |
|v_decile_score|Integer    |Specifically the score for "Violent Recidivism" risk.       |
|v_score_text  |Categorical|The interpreted risk level for the violent recidivism score.|
|in_custody    |Date/Time  |The date the defendant entered jail for the current charge. |
|out_custody   |Date/Time  |The date the defendant was released from jail.              |
|id            |Integer    |Unique identifier for the record in the ProPublica dataset. |
|name          |String     |Full name of the defendant.                                 |
|first / last  |String     |First and last name of the defendant (separated).           |
|dob           |Date       |Date of birth of the defendant.                             |
|compas_screening_date|Date       |The specific day the COMPAS assessment was administered.    |
|c_case_number |String     |The court case number for the current offense.              |
|c_offense_date|Date       |The date the current crime was allegedly committed.         |
|c_arrest_date |Date       |The date the defendant was taken into custody for the current charge.|
|c_days_from_compas|Integer    |Number of days between the current case and the COMPAS screening.|
|c_jail_in / out|DateTime   |The exact timestamps of when the defendant entered and left jail for the current charge.|
|c_case_number |String     |The court case number for the current offense.              |
|c_offense_date|Date       |The date the current crime was allegedly committed.         |
|c_arrest_date |Date       |The date the defendant was taken into custody for the current charge.|
|c_days_from_compas|Integer    |Number of days between the current case and the COMPAS screening.|
|c_jail_in / out|DateTime   |The exact timestamps of when the defendant entered and left jail for the current charge.|
|violent_recid |Numeric    |Often used as a null placeholder or sub-indicator for violent acts.|
|is_violent_recid|Binary     |Flag specifically for whether the new offense was violent (1 = Yes).|
|vr_case_number|String     |Case number for the violent recidivism event.               |
|vr_charge_degree|Categorical|Degree of the violent charge (F/M).                         |
|vr_offense_date|Date       |Date the violent crime occurred.                            |
|vr_charge_desc|String     |Description of the violent offense (e.g., "Battery").       |
|type_of_assessment|String     |Usually "Risk of Recidivism."                               |
|v_type_of_assessment|String     |Usually "Risk of Violence."                                 |
|screening_date|Date       |Duplicate of the COMPAS screening date.                     |
|start / end   |Integer    |Used for Survival Analysis (days from screening to either recidivism or end of study).|
|event         |Binary     |Used in Cox Proportional Hazards models to indicate if a recidivism "event" occurred.|


