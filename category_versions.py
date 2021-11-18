CAT_VERSIONS = {
    "versions": {
        "RACE_VERSION": 2,
        "GENDER_VERSION": 1,
        "AGE_VERSION": 4,
        "ETHNICITY_VERSION": 1,
        "PRIOR_VISITS_VERSION": 1,
        "VISIT_TYPE_VERSION": 1,
        "TRAUMA STATUS_VERSION": 1
    },
    "categories": {
        "RACE_VERSION_1": {
            "8657": "Native American", 
            "8515": "Asian", 
            "8516": "Black", 
            "8557": "Pacific Islander", 
            "8527": "White"
            },
        "RACE_VERSION_2": {
            "8515": "Asian",
            "8516": "Black",
            "8527": "White"
        },
        "ETHNICITY_VERSION_1": {
            "38003564": "Not Hispanic",
            "38003563": "Hispanic"
        },
        "GENDER_VERSION_1": {
            "8532": "Female", 
            "8507": "Male"
            },
        "AGE_VERSION_1": {
            tuple([0, 18]): "0 - 17",
            tuple([18, 35]): "18 - 34",
            tuple([35, 50]): "35 - 49",
            tuple([50, 70]): "50 - 69",
            tuple([70, 90]): "70 - 89",
            tuple([90, 150]): "90 +"
        },
        "AGE_VERSION_2": {
            tuple([0, 18]): "0 - 17",
            tuple([18, 65]): "18 - 64",
            tuple([65, 150]): "65 +"
        },
        "AGE_VERSION_3": {
            tuple([0, 10]): "0 - 10",
            tuple([11, 20]): "11 - 20",
            tuple([21, 30]): "21 - 30",
            tuple([31, 40]): "31 - 40",
            tuple([41, 50]): "41 - 50",
            tuple([51, 60]): "51 - 60",
            tuple([61, 70]): "61 - 70",
            tuple([71, 80]): "71 - 80",
            tuple([81, 90]): "81 - 90",
            tuple([91, 100]): "91 - 100",
            tuple([101, 200]): "100 +"
        },
        "AGE_VERSION_4": {
            tuple([0, 18]): "0 - 17",
            tuple([18, 35]): "18 - 34",
            tuple([35, 65]): "35 - 64",
            tuple([65, 100]): "65 - 99",
            tuple([100, 150]): "100 +"
        },
        "PRIOR_VISITS_VERSION_1": {
            tuple([0, 1]): "0",
            tuple([1, 11]): "1 - 10",
            tuple([11, 101]): "11 - 100",
            tuple([100, 6000]): "100 +"
        },
        "VISIT_TYPE_VERSION_1": {
            9201: "Inpatient Visit", 
            9202: "Outpatient Visit",
            9203: "Emergency Room Visit", 
        },
        "TRAUMA STATUS_VERSION_1": {
            "Trauma": "Trauma",
            "No Trauma": "No Trauma"
        }
    }
}
