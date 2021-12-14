from flask import Flask, request, render_template
from flask_cors import cross_origin
import sklearn
import pickle
import pandas as pd
from datetime import datetime
import json

app = Flask(__name__)
model = pickle.load(open("outcome_model_ct.pkl", "rb"))

# features
json_file = 'ml_features.json'

with open(json_file) as outfile:  
    features=json.load(outfile)

x_train=features['modeling']


@app.route("/")
@cross_origin()
def home():
    return render_template("home.html")




@app.route("/predict", methods = ["GET", "POST"])
@cross_origin()
def predict():
    if request.method == "POST":

        originalLoanAmount = float(request.form["LoanAmount"])
        originalLoanTerm = float(request.form["LoanTerm"])
        paymentToIncomePercentage = float(request.form["paymentToIncomePercentage"])/100
        originalInterestRatePercentage = float(request.form["InterestRatePercentage"])/100

        vehicleManufacturerName = request.form["vehicleManufacturerName"]
        vehicle_col=[x.split('_')[1] for x in x_train if x.startswith("vehicleManufacturerName")]
        Manufacturer={}
        for val in vehicle_col:
          if vehicleManufacturerName.lower().replace(' ','') == val:
            Manufacturer['vehicleManufacturerName_' + val.lower()]=1
          else:
            Manufacturer['vehicleManufacturerName_' + val.lower()]=0


        vehicleModelYear = float(request.form["vehicleModelYear"])

        vehicleNewUsedCodeM = request.form["vehicleNewUsedCodeM"]
        if vehicleNewUsedCodeM == 'New':
          vehicleNewUsedCodeM=1
        else:
          vehicleNewUsedCodeM=0

        vehicleValueAmount = float(request.form["vehicleValueAmount"])
        obligorCreditScore = float(request.form["obligorCreditScore"])

        vehicleValueSourceCodeM = request.form["vehicleValueSourceCodeM"]
        source_col=[x.split('_')[1] for x in x_train if x.startswith("vehicleValueSourceCodeM")]
        VehicleSource={}
        for val in source_col:
          if vehicleValueSourceCodeM.lower().replace(' ','') == val:
            VehicleSource['vehicleValueSourceCodeM_' + val.lower()]=1
          else:
            VehicleSource['vehicleValueSourceCodeM_' + val.lower()]=0

        obligorGeographicLocation = request.form["obligorGeographicLocation"]
        loc_col=[x.split('_')[1] for x in x_train if x.startswith("obligorGeographicLocation")]
        Location={}
        for val in loc_col:
          if obligorGeographicLocation.lower().replace(' ','') == val:
            Location['obligorGeographicLocation_' + val.lower()]=1
          else:
            Location['obligorGeographicLocation_' + val.lower()]=0

        underwritingIndicator = request.form["underwritingIndicator"]
        if underwritingIndicator== "True":
          underwritingIndicator=1
        else:
          underwritingIndicator=0

        obligorCreditScoreType = request.form["obligorCreditScoreType"]
        if obligorCreditScoreType=='Credit Bureau Score':
          obligorCreditScoreType=1
        else:
          obligorCreditScoreType=0

        obligorEmploymentVerificationCodeM = request.form["obligorEmploymentVerificationCodeM"]
        if obligorEmploymentVerificationCodeM== 'Stated, Not Verified':
          obligorEmploymentVerificationCodeM=1
        else:
          obligorEmploymentVerificationCodeM=0

        vehicleTypeCodeM = request.form["vehicleTypeCodeM"]
        type_col=[x.split('_')[1] for x in x_train if x.startswith("vehicleTypeCodeM")]
        Vehicletype={}
        for val in type_col:
            if vehicleTypeCodeM.lower() == val:
                Vehicletype['vehicleTypeCodeM_' + val.lower()]=1
            else:
                Vehicletype['vehicleTypeCodeM_' + val.lower()]=0

        obligorIncomeVerificationLevelCodeM = request.form["obligorIncomeVerificationLevelCodeM"]
        veri_level=[x.split('_')[1] for x in x_train if x.startswith("obligorIncomeVerificationLevelCodeM")]
        obligorIncome={}
        for val in veri_level:
          if obligorIncomeVerificationLevelCodeM.lower().replace(' ','') == val:
            obligorIncome['obligorIncomeVerificationLevelCodeM_' + val.lower()]=1
          else:
            obligorIncome['obligorIncomeVerificationLevelCodeM_' + val.lower()]=0

        ltv=originalLoanAmount/vehicleValueAmount

        coObligorIndicator = request.form["coObligorIndicator"]
        if coObligorIndicator == "True":
          coObligorIndicator=1
        else:
          coObligorIndicator=0

        eventYear=int(datetime.now().year)
        # AIR ASIA = 0 (not in column)
        
        data=[originalLoanAmount,originalLoanTerm,paymentToIncomePercentage,originalInterestRatePercentage,vehicleNewUsedCodeM,
        vehicleModelYear,vehicleValueAmount,coObligorIndicator,obligorCreditScore,underwritingIndicator,obligorCreditScoreType,
        obligorEmploymentVerificationCodeM,ltv,eventYear,Manufacturer['vehicleManufacturerName_alfaromeo'],Manufacturer['vehicleManufacturerName_audi'],
        Manufacturer['vehicleManufacturerName_bentley'],Manufacturer['vehicleManufacturerName_bmw'],Manufacturer['vehicleManufacturerName_buick'],
        Manufacturer['vehicleManufacturerName_cadillac'],Manufacturer['vehicleManufacturerName_chevrolet'],
        Manufacturer['vehicleManufacturerName_chrysler'],Manufacturer['vehicleManufacturerName_dodge'],Manufacturer['vehicleManufacturerName_fiat'],
        Manufacturer['vehicleManufacturerName_ford'],Manufacturer['vehicleManufacturerName_genesis'],Manufacturer['vehicleManufacturerName_gmc'],
        Manufacturer['vehicleManufacturerName_honda'],Manufacturer['vehicleManufacturerName_hummer'],
        Manufacturer['vehicleManufacturerName_hyundai'],Manufacturer['vehicleManufacturerName_infiniti'],Manufacturer['vehicleManufacturerName_jaguar'],
        Manufacturer['vehicleManufacturerName_jeep'],Manufacturer['vehicleManufacturerName_kia'],Manufacturer['vehicleManufacturerName_landrover'],
        Manufacturer['vehicleManufacturerName_lexus'],
        Manufacturer['vehicleManufacturerName_lincoln'],Manufacturer['vehicleManufacturerName_maserati'],Manufacturer['vehicleManufacturerName_mazda'],
        Manufacturer['vehicleManufacturerName_mercedes-benz'],
        Manufacturer['vehicleManufacturerName_mercury'],Manufacturer['vehicleManufacturerName_mini'],Manufacturer['vehicleManufacturerName_mitsubishi'],
        Manufacturer['vehicleManufacturerName_nissan'],Manufacturer['vehicleManufacturerName_pontiac'],Manufacturer['vehicleManufacturerName_porsche'],
        Manufacturer['vehicleManufacturerName_ram'],
        Manufacturer['vehicleManufacturerName_saab'],Manufacturer['vehicleManufacturerName_saturn'],Manufacturer['vehicleManufacturerName_smart'],
        Manufacturer['vehicleManufacturerName_subaru'],
        Manufacturer['vehicleManufacturerName_suzuki'],Manufacturer['vehicleManufacturerName_tesla'],Manufacturer['vehicleManufacturerName_toyota'],
        Manufacturer['vehicleManufacturerName_volkswagen'],
        Manufacturer['vehicleManufacturerName_volvo'],VehicleSource['vehicleValueSourceCodeM_kellybluebook'],VehicleSource['vehicleValueSourceCodeM_other'],
        Location['obligorGeographicLocation_ak'],
        Location['obligorGeographicLocation_al'],Location['obligorGeographicLocation_ap'],Location['obligorGeographicLocation_ar'],
        Location['obligorGeographicLocation_az'],Location['obligorGeographicLocation_ca'],Location['obligorGeographicLocation_co'],
        Location['obligorGeographicLocation_ct'],Location['obligorGeographicLocation_dc'],
        Location['obligorGeographicLocation_de'],Location['obligorGeographicLocation_fl'],Location['obligorGeographicLocation_ga'],Location['obligorGeographicLocation_gu'],
        Location['obligorGeographicLocation_hi'],Location['obligorGeographicLocation_ia'],Location['obligorGeographicLocation_id'],Location['obligorGeographicLocation_il'],
        Location['obligorGeographicLocation_in'],Location['obligorGeographicLocation_ks'],Location['obligorGeographicLocation_ky'],Location['obligorGeographicLocation_la'],
        Location['obligorGeographicLocation_ma'],Location['obligorGeographicLocation_md'],Location['obligorGeographicLocation_me'],Location['obligorGeographicLocation_mi'],
        Location['obligorGeographicLocation_mn'],Location['obligorGeographicLocation_mo'],Location['obligorGeographicLocation_ms'],Location['obligorGeographicLocation_mt'],
        Location['obligorGeographicLocation_nc'],Location['obligorGeographicLocation_nd'],Location['obligorGeographicLocation_ne'],Location['obligorGeographicLocation_nh'],
        Location['obligorGeographicLocation_nj'],Location['obligorGeographicLocation_nm'],Location['obligorGeographicLocation_nv'],Location['obligorGeographicLocation_ny'],
        Location['obligorGeographicLocation_oh'],Location['obligorGeographicLocation_ok'],Location['obligorGeographicLocation_or'],Location['obligorGeographicLocation_pa'],
        Location['obligorGeographicLocation_pr'],Location['obligorGeographicLocation_ri'],Location['obligorGeographicLocation_sc'],Location['obligorGeographicLocation_sd'],
        Location['obligorGeographicLocation_tn'],Location['obligorGeographicLocation_tx'],Location['obligorGeographicLocation_ut'],Location['obligorGeographicLocation_va'],
        Location['obligorGeographicLocation_vi'],Location['obligorGeographicLocation_vt'],Location['obligorGeographicLocation_wa'],Location['obligorGeographicLocation_wi'],
        Location['obligorGeographicLocation_wv'],Location['obligorGeographicLocation_wy'],Vehicletype['vehicleTypeCodeM_suv'],Vehicletype['vehicleTypeCodeM_truck'],
        Vehicletype['vehicleTypeCodeM_unavailable'],obligorIncome['obligorIncomeVerificationLevelCodeM_stated,notverified'],
        obligorIncome['obligorIncomeVerificationLevelCodeM_stated,verifiedbutnottolevel4orlevel5']]
        
        prediction=model.predict(data)

        if prediction == 0:
            return render_template('home.html',prediction_text="Congrats!, Your loan will not default!")
        else:
            return render_template('home.html',prediction_text="Sorry, Your loan will default!")

    return render_template("home.html")




if __name__ == "__main__":
    app.run(debug=True)
