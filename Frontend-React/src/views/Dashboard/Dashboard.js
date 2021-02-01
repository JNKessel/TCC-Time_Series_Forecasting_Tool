/*!

=========================================================
* Now UI Dashboard PRO React - v1.4.0
=========================================================

* Product Page: https://www.creative-tim.com/product/now-ui-dashboard-pro-react
* Copyright 2020 Creative Tim (https://www.creative-tim.com)

* Coded by Creative Tim

=========================================================

* The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

*/
import React, { useState, useRef, useEffect } from 'react'
// react plugin used to create charts
import { Line, Bar } from 'react-chartjs-2'
// react plugin for creating vector maps
import { VectorMap } from 'react-jvectormap'
// react component used to create sweet alerts
import SweetAlert from 'react-bootstrap-sweetalert'

// reactstrap components
import {
  Card,
  CardHeader,
  CardBody,
  CardFooter,
  CardTitle,
  DropdownToggle,
  DropdownMenu,
  DropdownItem,
  UncontrolledDropdown,
  Table,
  Progress,
  Row,
  Col,
  Input,
  FormGroup,
  Button,
  Modal,
  ModalBody,
  ModalFooter,
  ModalHeader,
  Label,
} from 'reactstrap'
import { BsFullscreen, BsFullscreenExit } from 'react-icons/bs'

// core components
import PanelHeader from 'components/PanelHeader/PanelHeader.js'

import Select from 'react-select'

import {
  dashboardPanelChart,
  BuildDashboardPanelChart,
  BuildDashboardPanelChartDiferentiation,
  BuildDashboardChartPredictions,
  BuildDashboardChartMetrics,
  dashboardActiveUsersChart,
  dashboardSummerChart,
  dashboardActiveCountriesCard,
} from 'variables/charts.js'

import jacket from 'assets/img/saint-laurent.jpg'
import shirt from 'assets/img/balmain.jpg'
import swim from 'assets/img/prada.jpg'

import { table_data } from 'variables/general.js'

import { ExcelRenderer, OutTable } from 'react-excel-renderer'

var mapData = {
  AU: 760,
  BR: 550,
  CA: 120,
  DE: 1300,
  FR: 540,
  GB: 690,
  GE: 200,
  IN: 200,
  RO: 600,
  RU: 300,
  US: 2920,
}

function Dashboard(props) {
  const uploadEl = useRef(null)
  const [sidebarMini, setSidebarMini] = useState(true)
  const [backgroundColor, setBackgroundColor] = useState('blue')
  const [dashboardHeaderSize, setDashboardHeaderSize] = useState('lg')

  //DASHBOARD STATES
  const mainDashChart = BuildDashboardPanelChart(
    [
      'JAN',
      'FEB',
      'MAR',
      'APR',
      'MAY',
      'JUN',
      'JUL',
      'AUG',
      'SEP',
      'OCT',
      'NOV',
      'DEC',
    ],
    [50, 150, 100, 190, 130, 90, 150, 160, 120, 140, 190, 95],
    190,
    50
  )

  const dashChartDiff = BuildDashboardPanelChartDiferentiation(
    [
      'JAN',
      'FEB',
      'MAR',
      'APR',
      'MAY',
      'JUN',
      'JUL',
      'AUG',
      'SEP',
      'OCT',
      'NOV',
      'DEC',
    ],
    [50, 150, 100, 190, 130, 90, 150, 160, 120, 140, 190, 95],
    190,
    50
  )

  const [file, setFile] = useState(null)
  const [fileName, setFileName] = useState(null)
  const [fileColumns, setFileColumns] = useState(null)
  const [fileRows, setFileRows] = useState(null)

  const [showFileUploadModal, setShowFileUploadModal] = useState(true)
  const [showFileUploadModalSuccess, setShowFileUploadModalSuccess] = useState(
    false
  )

  const [
    dashboardHeaderTimeseriesMaxValue,
    setDashboardHeaderTimeseriesMaxValue,
  ] = useState(190)
  const [
    dashboardHeaderTimeseriesAvarageValue,
    setDashboardHeaderTimeseriesAvarageValue,
  ] = useState(190)
  const [
    dashboardHeaderTimeseriesMedianValue,
    setDashboardHeaderTimeseriesMedianValue,
  ] = useState(190)
  const [
    dashboardHeaderTimeseriesMinValue,
    setDashboardHeaderTimeseriesMinValue,
  ] = useState(50)
  const [mainDashBoardPanelChart, setMainDashBoardPanelChart] = useState(
    mainDashChart
  )
  const [
    dashboardPanelChartDiferentiation,
    setDashboardPanelChartDiferentiation,
  ] = useState(dashChartDiff)

  const [dashboardChartPredictions, setDashboardChartPredictions] = useState({})

  const [
    dashboardChartPredictionsValidation,
    setDashboardChartPredictionsValidation,
  ] = useState({})

  const [
    dashboardChartPredictionsTraining,
    setDashboardChartPredictionsTraining,
  ] = useState({})

  const [seriesData, setSeriesData] = useState([])

  const [seriesTimestamps, setSeriesTimestamps] = useState([])

  const [trainningTouched, setTrainningTouched] = useState(false)

  const [modelsMetrics, setModelsMetrics] = useState({
    METRICS: {
      MAPE: { TRAINING: {}, VALIDATION: {}, TEST: {} },
      MAE: { TRAINING: {}, VALIDATION: {}, TEST: {} },
    },
  })

  const [trainingResults, setTrainingResults] = useState({
    PRED: { TRAINING: {}, VALIDATION: {}, TEST: {} },
  })

  const [dashboardChartMetrics, setDashboardChartMetrics] = useState({})

  const [dashboardChartMetricsMAE, setDashboardChartMetricsMAE] = useState({})

  const [selectedModels, setSelectedModels] = useState([])

  const [predictionChartFullSize, setPredictionChartFullSize] = useState(false)

  const [isSettingsModalOpen, setIsSettingsModalOpen] = useState(false)

  const [datasetSplitPercentages, setDatasetSplitPercentages] = useState([
    0.7,
    0.2,
    0.1,
  ])

  const [numberOfEpochs, setNumberOfEpochs] = useState(500)

  const [denseModelHiddenLayers, setDenseModelHiddenLayers] = useState([64, 64])

  const [
    multiStepDenseModelHiddenLayers,
    setMultiStepDenseModelHiddenLayers,
  ] = useState([32, 32])

  const [
    multiStepDenseModelInputSteps,
    setMultiStepDenseModelInputSteps,
  ] = useState(3)

  const [lstmModelWindowSize, setLstmModelWindowSize] = useState(12)

  const [selectedModelsArray, setSelectedModelsArray] = useState([])

  const addFile = (e) => {
    let _fileName = e.target.files[0].name
    let _file = e.target.files[0]

    //just pass the fileObj as parameter
    ExcelRenderer(_file, (err, resp) => {
      if (err) {
        console.log(err)
      } else {
        setFileColumns(resp.cols)
        setFileRows(resp.rows)

        setFileName(_fileName)
        setFile(_file)
      }
    })
  }

  const createTableData = () => {
    var tableRows = []
    for (var i = 0; i < table_data.length; i++) {
      tableRows.push(
        <tr key={i}>
          <td>
            <div className='flag'>
              <img src={table_data[i].flag} alt='us_flag' />
            </div>
          </td>
          <td>{table_data[i].country}</td>
          <td className='text-right'>{table_data[i].count}</td>
          <td className='text-right'>{table_data[i].percentage}</td>
        </tr>
      )
    }
    return tableRows
  }

  const trainModels = (callback) => {
    selectedModels.forEach((element) => {
      if (element.value === 'baseline') {
        train_model('baseline', 'Baseline')
      }
      if (element.value === 'linear') {
        train_model('linear', 'Linear model')
      }
      if (element.value === 'dense') {
        train_model('dense', 'Dense model')
      }
      if (element.value === 'multistepdense') {
        train_model('multistepdense', 'Multi step dense model')
      }
      if (element.value === 'convolutional') {
        train_model('convolutional', 'Convolutional')
      }
      if (element.value === 'recurrentlstm') {
        train_model('recurrentlstm', 'Recurrent LSTM')
      }
    })
  }

  const train_model = (urlSufix, modelName) => {
    var xmlHttp = new XMLHttpRequest()
    xmlHttp.onreadystatechange = function () {
      if (xmlHttp.readyState === 4 && xmlHttp.status === 200) {
        let response_object = JSON.parse(xmlHttp.responseText)

        let tempTrainingResults = {
          ...trainingResults,
        }
        tempTrainingResults['PRED']['TRAINING'][modelName] =
          response_object.training_data_predictions
        tempTrainingResults['PRED']['VALIDATION'][modelName] =
          response_object.validation_data_predictions
        tempTrainingResults['PRED']['TEST'][modelName] =
          response_object.test_data_predictions

        setTrainingResults(tempTrainingResults)

        //CALC MAPE FOR TEST DATA
        let seriesTestData = seriesData.slice(
          seriesData.length *
            (datasetSplitPercentages[0] + datasetSplitPercentages[1])
        )
        seriesTestData = seriesTestData.slice(
          seriesTestData.length - response_object.test_data_predictions.length
        )
        const mapeMetricTest = calcMAPE(
          seriesTestData,
          response_object.test_data_predictions
        )

        //CALC MAE  FOR TEST DATA
        const maeMetricTest = calcMAE(
          seriesTestData,
          response_object.test_data_predictions
        )

        //CALC MAPE FOR VALIDATION DATA
        let seriesValidationData = seriesData.slice(
          seriesData.length * datasetSplitPercentages[0],
          seriesData.length *
            (datasetSplitPercentages[0] + datasetSplitPercentages[1])
        )
        seriesValidationData = seriesValidationData.slice(
          seriesValidationData.length -
            response_object.validation_data_predictions.length
        )
        const mapeMetricValidation = calcMAPE(
          seriesValidationData,
          response_object.validation_data_predictions
        )

        //CALC MAE FOR VALIDATION DATA
        const maeMetricValidation = calcMAE(
          seriesValidationData,
          response_object.validation_data_predictions
        )

        //CALC MAPE FOR TRAINING DATA
        let seriesTrainingData = seriesData.slice(
          0,
          seriesData.length * datasetSplitPercentages[0]
        )
        seriesTrainingData = seriesTrainingData.slice(
          seriesTrainingData.length -
            response_object.training_data_predictions.length
        )
        const mapeMetricTraining = calcMAPE(
          seriesTrainingData,
          response_object.training_data_predictions
        )

        //CALC MAE FOR TRAINING DATA
        const maeMetricTraining = calcMAE(
          seriesTrainingData,
          response_object.training_data_predictions
        )

        let tempModelMetrics = {
          ...modelsMetrics,
        }

        //MAPE
        tempModelMetrics['METRICS']['MAPE']['TRAINING'][
          modelName
        ] = mapeMetricTraining
        tempModelMetrics['METRICS']['MAPE']['VALIDATION'][
          modelName
        ] = mapeMetricValidation
        tempModelMetrics['METRICS']['MAPE']['TEST'][modelName] = mapeMetricTest

        //MAE
        tempModelMetrics['METRICS']['MAE']['TRAINING'][
          modelName
        ] = maeMetricTraining
        tempModelMetrics['METRICS']['MAE']['VALIDATION'][
          modelName
        ] = maeMetricValidation
        tempModelMetrics['METRICS']['MAE']['TEST'][modelName] = maeMetricTest

        setModelsMetrics(tempModelMetrics)

        setDashboardChartPredictions(
          BuildDashboardChartPredictions(
            seriesTimestamps.slice(
              seriesData.length *
                (datasetSplitPercentages[0] + datasetSplitPercentages[1])
            ),
            seriesData.slice(
              seriesData.length *
                (datasetSplitPercentages[0] + datasetSplitPercentages[1])
            ),
            tempTrainingResults.PRED.TEST,
            Math.max(
              ...seriesData.slice(
                seriesData.length *
                  (datasetSplitPercentages[0] + datasetSplitPercentages[1])
              )
            ),
            Math.min(
              ...seriesData.slice(
                seriesData.length *
                  (datasetSplitPercentages[0] + datasetSplitPercentages[1])
              )
            )
          )
        )

        setDashboardChartPredictionsValidation(
          BuildDashboardChartPredictions(
            seriesTimestamps.slice(
              seriesData.length * datasetSplitPercentages[0],
              seriesData.length *
                (datasetSplitPercentages[0] + datasetSplitPercentages[1])
            ),
            seriesData.slice(
              seriesData.length * datasetSplitPercentages[0],
              seriesData.length *
                (datasetSplitPercentages[0] + datasetSplitPercentages[1])
            ),
            tempTrainingResults.PRED.VALIDATION,
            Math.max(
              ...seriesData.slice(
                seriesData.length * datasetSplitPercentages[0],
                seriesData.length *
                  (datasetSplitPercentages[0] + datasetSplitPercentages[1])
              )
            ),
            Math.min(
              ...seriesData.slice(
                seriesData.length * datasetSplitPercentages[0],
                seriesData.length *
                  (datasetSplitPercentages[0] + datasetSplitPercentages[1])
              )
            )
          )
        )

        setDashboardChartPredictionsTraining(
          BuildDashboardChartPredictions(
            seriesTimestamps.slice(
              0,
              seriesData.length * datasetSplitPercentages[0]
            ),
            seriesData.slice(0, seriesData.length * datasetSplitPercentages[0]),
            tempTrainingResults.PRED.TRAINING,
            Math.max(
              ...seriesData.slice(
                0,
                seriesData.length * datasetSplitPercentages[0]
              )
            ),
            Math.min(
              ...seriesData.slice(
                0,
                seriesData.length * datasetSplitPercentages[0]
              )
            )
          )
        )

        //MAPE
        let metricValuesArray = []
        let maxMetricValue = 0
        for (const [key, value] of Object.entries(
          tempModelMetrics.METRICS.MAPE.TEST
        )) {
          metricValuesArray.push(value)
        }
        metricValuesArray.forEach(function (el) {
          if (el > maxMetricValue) {
            maxMetricValue = el
          }
        })

        //MAE
        let metricValuesArrayMAE = []
        let maxMetricValueMAE = 0
        for (const [key, value] of Object.entries(
          tempModelMetrics.METRICS.MAE.TEST
        )) {
          metricValuesArrayMAE.push(value)
        }
        metricValuesArrayMAE.forEach(function (el) {
          if (el > maxMetricValueMAE) {
            maxMetricValueMAE = el
          }
        })

        setDashboardChartMetrics(
          BuildDashboardChartMetrics(
            tempModelMetrics.METRICS.MAPE,
            maxMetricValue
          )
        )

        setDashboardChartMetricsMAE(
          BuildDashboardChartMetrics(
            tempModelMetrics.METRICS.MAE,
            maxMetricValueMAE
          )
        )

        if (
          Object.keys(tempTrainingResults['PRED']['TRAINING']).length ===
          selectedModels.length
        ) {
          setTrainingResults({
            PRED: { TRAINING: {}, VALIDATION: {}, TEST: {} },
          })
          setModelsMetrics({
            METRICS: {
              MAPE: { TRAINING: {}, VALIDATION: {}, TEST: {} },
              MAE: { TRAINING: {}, VALIDATION: {}, TEST: {} },
            },
          })
        }

        setTrainningTouched(true)
      }
    }
    let specificContect = {}
    if (urlSufix === 'dense') {
      specificContect['HiddenLayers'] = denseModelHiddenLayers
    } else if (urlSufix === 'multistepdense') {
      specificContect['HiddenLayers'] = multiStepDenseModelHiddenLayers
      specificContect['InputSteps'] = multiStepDenseModelInputSteps
    } else if (urlSufix === 'recurrentlstm') {
      specificContect['WindowSize'] = lstmModelWindowSize
    }
    xmlHttp.open('POST', 'http://localhost:5000/' + urlSufix, true) // true for asynchronous
    xmlHttp.setRequestHeader('Content-Type', 'application/json;charset=UTF-8')
    xmlHttp.send(
      JSON.stringify({
        seriesData: seriesData,
        datasetPercentages: datasetSplitPercentages,
        numberOfEpochs: numberOfEpochs,
        specificContect: specificContect,
      })
    )
  }

  const calcMAPE = (true_values, pedictions) => {
    let mape = 0
    for (let i = 0; i < true_values.length; i++) {
      mape = mape + Math.abs((true_values[i] - pedictions[i]) / true_values[i])
    }
    mape = mape / true_values.length

    return mape
  }

  const calcMAE = (true_values, pedictions) => {
    let mae = 0
    for (let i = 0; i < true_values.length; i++) {
      mae = mae + Math.abs(true_values[i] - pedictions[i])
    }
    mae = mae / true_values.length

    return mae
  }

  return (
    <>
      {showFileUploadModal && (
        <SweetAlert
          style={{ display: 'block', marginTop: '-100px' }}
          title='Upload a sheet with time series data'
          onConfirm={() => {
            if (file) {
              let newData = []
              let newTimestamps = []
              let newDataDiff = []
              let previouValue = 0
              let maxValue = fileRows[1][1]
              let minValue = fileRows[1][1]
              let maxValueDiff = -1000
              let minValueDiff = 1000
              let sum = 0
              let countRows = 0
              fileRows.forEach((el, index) => {
                if (el.length > 0 && index > 0) {
                  newTimestamps.push(el[0])
                  newData.push(el[1])

                  sum = sum + el[1]
                  countRows = countRows + 1

                  if (el[1] > maxValue) {
                    maxValue = el[1]
                  }

                  if (el[1] < minValue) {
                    minValue = el[1]
                  }

                  if (index > 1) {
                    let diff = el[1] - previouValue
                    newDataDiff.push(diff)

                    if (diff > maxValueDiff) {
                      maxValueDiff = diff
                    }

                    if (diff < minValueDiff) {
                      minValueDiff = diff
                    }
                  }

                  previouValue = el[1]
                }
              })

              setShowFileUploadModal(false)
              setShowFileUploadModalSuccess(true)

              setTimeout(() => {
                setShowFileUploadModalSuccess(false)

                setSeriesData(newData)
                setSeriesTimestamps(newTimestamps)

                setMainDashBoardPanelChart(
                  BuildDashboardPanelChart(
                    newTimestamps,
                    newData,
                    maxValue,
                    minValue
                  )
                )

                setDashboardPanelChartDiferentiation(
                  BuildDashboardPanelChartDiferentiation(
                    newTimestamps.slice(1),
                    newDataDiff,
                    maxValueDiff,
                    minValueDiff
                  )
                )

                setDashboardHeaderTimeseriesMaxValue(maxValue)
                setDashboardHeaderTimeseriesMinValue(minValue)
                setDashboardHeaderTimeseriesAvarageValue(sum / countRows)
                setDashboardHeaderTimeseriesMedianValue(
                  fileRows[Math.round(countRows / 2)][1]
                )
              }, 1600)
            }
          }}
          confirmBtnBsStyle='info'
        >
          <FormGroup className='form-file-upload form-file-simple'>
            <Input
              type='text'
              className='inputFileVisible'
              placeholder='File chooser...'
              onClick={(e) => uploadEl.current.click(e)}
              defaultValue={fileName}
            />
            <input
              type='file'
              className='inputFileHidden'
              style={{ zIndex: -1 }}
              ref={uploadEl}
              onChange={(e) => addFile(e)}
            />
          </FormGroup>
        </SweetAlert>
      )}
      {showFileUploadModalSuccess && (
        <SweetAlert
          success
          style={{ display: 'block', marginTop: '-100px' }}
          title='Success!'
          showConfirm={false}
          onConfirm={() => {}}
        >
          Your file was uploaded.
        </SweetAlert>
      )}
      <PanelHeader
        size={dashboardHeaderSize}
        content={
          <>
            <UncontrolledDropdown
              style={{
                position: 'absolute',
                right: 0,
                marginRight: '40px',
                marginTop: '-76px',
                zIndex: '10000',
              }}
            >
              <DropdownToggle
                className='btn-round btn-icon'
                color='default'
                outline
                style={{ color: 'white', fontSize: '16px' }}
                onClick={() => {
                  if (dashboardHeaderSize === 'lg') {
                    setDashboardHeaderSize('fs')
                  } else {
                    setDashboardHeaderSize('lg')
                  }
                }}
              >
                <div style={{ marginTop: '-3px' }}>
                  {dashboardHeaderSize === 'lg' ? (
                    <BsFullscreen />
                  ) : (
                    <BsFullscreenExit />
                  )}
                </div>
              </DropdownToggle>
            </UncontrolledDropdown>
            <Line
              data={mainDashBoardPanelChart.data}
              options={mainDashBoardPanelChart.options}
            />
          </>
        }
      />
      <div className='content'>
        <Row>
          <Col xs={12} md={12}>
            <Card className='card-stats card-raised'>
              <CardBody>
                <Row>
                  <Col md='3'>
                    <div className='statistics'>
                      <div className='info'>
                        <div className='icon icon-primary'>
                          <i className='now-ui-icons ui-2_chat-round' />
                        </div>
                        <h3 className='info-title'>
                          {dashboardHeaderTimeseriesMaxValue.toFixed(2)}
                        </h3>
                        <h6 className='stats-title'>Max value</h6>
                      </div>
                    </div>
                  </Col>
                  <Col md='3'>
                    <div className='statistics'>
                      <div className='info'>
                        <div className='icon icon-success'>
                          <i className='now-ui-icons business_money-coins' />
                        </div>
                        <h3 className='info-title'>
                          {dashboardHeaderTimeseriesMinValue.toFixed(2)}
                        </h3>
                        <h6 className='stats-title'>Min value</h6>
                      </div>
                    </div>
                  </Col>
                  <Col md='3'>
                    <div className='statistics'>
                      <div className='info'>
                        <div className='icon icon-info'>
                          <i className='now-ui-icons users_single-02' />
                        </div>
                        <h3 className='info-title'>
                          {dashboardHeaderTimeseriesAvarageValue.toFixed(2)}
                        </h3>
                        <h6 className='stats-title'>Average value</h6>
                      </div>
                    </div>
                  </Col>
                  <Col md='3'>
                    <div className='statistics'>
                      <div className='info'>
                        <div className='icon icon-danger'>
                          <i className='now-ui-icons objects_support-17' />
                        </div>
                        <h3 className='info-title'>
                          {dashboardHeaderTimeseriesMedianValue.toFixed(2)}
                        </h3>
                        <h6 className='stats-title'>Median value</h6>
                      </div>
                    </div>
                  </Col>
                </Row>
              </CardBody>
            </Card>
          </Col>
        </Row>
        <Row>
          <Col xs={12} md={12}>
            <Card className='card-chart'>
              <CardHeader>
                <h5 className='card-category'>Series diferentiation</h5>
              </CardHeader>
              <CardBody>
                <div className='chart-area-lg'>
                  <Line
                    data={dashboardPanelChartDiferentiation.data}
                    options={dashboardPanelChartDiferentiation.options}
                  />
                </div>
              </CardBody>
            </Card>
          </Col>
        </Row>
        <Row>
          <Col xs={12} md={12}>
            <Card className='card-chart'>
              <CardHeader>
                <h5 className='card-category'>Model training</h5>
              </CardHeader>
              <CardBody>
                <div className='chart-area-lg'>
                  <Row>
                    <Col
                      xs={12}
                      md={6}
                      style={{
                        margin: '0 auto',
                        textAlign: 'center',
                      }}
                    >
                      <CardTitle tag='h4'>Training models</CardTitle>
                      <Row>
                        <Col
                          xs={9}
                          md={9}
                          style={{
                            margin: '0 auto',
                            textAlign: 'start',
                          }}
                        >
                          <Select
                            className='react-select warning'
                            classNamePrefix='react-select'
                            isMulti
                            closeMenuOnSelect={false}
                            placeholder='Multiple Select'
                            name='multipleSelect'
                            options={[
                              {
                                value: 'baseline',
                                label: 'Baseline',
                              },
                              {
                                value: 'linear',
                                label: 'Linear model',
                              },
                              {
                                value: 'dense',
                                label: 'Dense model',
                              },
                              {
                                value: 'multistepdense',
                                label: 'Multi step dense model',
                              },
                              {
                                value: 'convolutional',
                                label: 'Convolutional',
                              },
                              {
                                value: 'recurrentlstm',
                                label: 'Recurrent LSTM',
                              },
                            ]}
                            value={selectedModels}
                            onChange={(value) => {
                              let temp = []
                              if (value) {
                                value.forEach((element) => {
                                  temp.push(element.value)
                                })
                              }
                              setSelectedModelsArray(temp)
                              setSelectedModels(value)
                            }}
                          />
                        </Col>
                      </Row>
                      <Button
                        color='primary'
                        className='btn-round'
                        style={{ marginTop: '30px' }}
                        onClick={() => {
                          setIsSettingsModalOpen(true)
                          setDatasetSplitPercentages(datasetSplitPercentages)
                        }}
                      >
                        Settings
                      </Button>
                      <Modal
                        isOpen={isSettingsModalOpen}
                        toggle={() =>
                          setIsSettingsModalOpen(!isSettingsModalOpen)
                        }
                        className='text-center'
                        style={{ maxWidth: '70%' }}
                      >
                        <ModalHeader
                          className='justify-content-center uppercase title'
                          toggle={() =>
                            setIsSettingsModalOpen(!isSettingsModalOpen)
                          }
                          tag='h4'
                        >
                          SETTINGS
                        </ModalHeader>
                        <ModalBody>
                          <Row style={{ textAlign: 'center' }}>
                            <Col xs={12} sm={3} md='12'>
                              <label
                                style={{
                                  fontSize: '14px',
                                  fontWeight: 'bolder',
                                  textDecoration: 'underline',
                                }}
                              >
                                General:
                              </label>
                            </Col>
                          </Row>
                          <Row style={{ textAlign: 'start' }}>
                            <Col xs={12} sm={3} md='4'>
                              <label>Train set percentage</label>
                              <Input
                                type='text'
                                onChange={(e) => {
                                  setDatasetSplitPercentages([
                                    e.target.value,
                                    datasetSplitPercentages[1],
                                    datasetSplitPercentages[2],
                                  ])
                                }}
                                placeholder='0.7'
                                value={datasetSplitPercentages[0]}
                              />
                            </Col>
                            <Col xs={12} sm={3} md='4'>
                              <label>Validation set percentage</label>
                              <Input
                                type='text'
                                onChange={(e) => {
                                  setDatasetSplitPercentages([
                                    datasetSplitPercentages[0],
                                    e.target.value,
                                    datasetSplitPercentages[2],
                                  ])
                                }}
                                value={datasetSplitPercentages[1]}
                              />
                            </Col>
                            <Col xs={12} sm={3} md='4'>
                              <label>Test set percentage</label>
                              <Input
                                type='text'
                                onChange={(e) => {
                                  setDatasetSplitPercentages([
                                    datasetSplitPercentages[0],
                                    datasetSplitPercentages[1],
                                    e.target.value,
                                  ])
                                }}
                                placeholder='0.1'
                                value={datasetSplitPercentages[2]}
                              />
                            </Col>
                          </Row>
                          <Row style={{ textAlign: 'start' }}>
                            <Col xs={12} sm={3} md='4'>
                              <label>Number of epochs: </label>
                              <Input
                                type='text'
                                onChange={(e) => {
                                  setNumberOfEpochs(
                                    parseInt(e.target.value)
                                      ? parseInt(e.target.value)
                                      : 0
                                  )
                                }}
                                placeholder='500'
                                value={numberOfEpochs}
                              />
                            </Col>
                          </Row>
                          {selectedModelsArray.includes('dense') && (
                            <>
                              <Row
                                style={{
                                  textAlign: 'center',
                                  marginTop: '40px',
                                }}
                              >
                                <Col xs={12} sm={3} md='12'>
                                  <label
                                    style={{
                                      fontSize: '14px',
                                      fontWeight: 'bolder',
                                      textDecoration: 'underline',
                                    }}
                                  >
                                    Dense model:
                                  </label>
                                </Col>
                              </Row>
                              <Row style={{ textAlign: 'start' }}>
                                {denseModelHiddenLayers.map((value, index) => {
                                  return (
                                    <>
                                      <Col
                                        xs={12}
                                        sm={3}
                                        md='3'
                                        style={{ maxWidth: '210px' }}
                                      >
                                        <label>
                                          Number of neurons Hidden Layer #
                                          {index + 1}
                                        </label>
                                        <Input
                                          type='text'
                                          value={value}
                                          onChange={(e) => {
                                            let temp = []
                                            for (const element of denseModelHiddenLayers) {
                                              temp.push(element)
                                            }
                                            temp[index] = parseFloat(
                                              e.target.value
                                            )
                                              ? parseFloat(e.target.value)
                                              : 0
                                            setDenseModelHiddenLayers(temp)
                                          }}
                                        />
                                      </Col>
                                      <Label
                                        sm={3}
                                        className='label-on-right'
                                        style={{ flex: '0' }}
                                        onClick={() => {
                                          let temp = []
                                          for (const element of denseModelHiddenLayers) {
                                            temp.push(element)
                                          }
                                          temp.splice(index, 1)
                                          setDenseModelHiddenLayers(temp)
                                        }}
                                      >
                                        <div
                                          style={{
                                            marginTop: '57px',
                                            cursor: 'pointer',
                                            color: 'red',
                                            marginLeft: '-23px',
                                          }}
                                        >
                                          <i className='now-ui-icons ui-1_simple-remove' />
                                        </div>
                                      </Label>
                                    </>
                                  )
                                })}
                                <Col
                                  xs={12}
                                  sm={3}
                                  md='3'
                                  style={{
                                    maxWidth: '120px',
                                    marginTop: '22px',
                                  }}
                                >
                                  <Button
                                    color='info'
                                    className='btn-round'
                                    onClick={() => {
                                      let temp = []
                                      for (const element of denseModelHiddenLayers) {
                                        temp.push(element)
                                      }
                                      temp.push(64)
                                      setDenseModelHiddenLayers(temp)
                                    }}
                                    style={{ marginTop: '30px' }}
                                  >
                                    Add{' '}
                                    <i className='now-ui-icons ui-1_simple-add' />
                                  </Button>
                                </Col>
                              </Row>
                            </>
                          )}
                          {selectedModelsArray.includes('multistepdense') && (
                            <>
                              {' '}
                              <Row
                                style={{
                                  textAlign: 'center',
                                  marginTop: '40px',
                                }}
                              >
                                <Col xs={12} sm={3} md='12'>
                                  <label
                                    style={{
                                      fontSize: '14px',
                                      fontWeight: 'bolder',
                                      textDecoration: 'underline',
                                    }}
                                  >
                                    Multi step dense model:
                                  </label>
                                </Col>
                              </Row>
                              <Row
                                style={{
                                  textAlign: 'start',
                                  marginTop: '0px',
                                  paddingBottom: '15px',
                                }}
                              >
                                <Col
                                  xs={12}
                                  sm={3}
                                  md='3'
                                  style={{ maxWidth: '210px' }}
                                >
                                  <label>Number of input steps</label>
                                  <Input
                                    type='text'
                                    onChange={(e) => {
                                      setMultiStepDenseModelInputSteps(
                                        parseFloat(e.target.value)
                                          ? parseFloat(e.target.value)
                                          : 0
                                      )
                                    }}
                                    placeholder='0.7'
                                    value={multiStepDenseModelInputSteps}
                                  />
                                </Col>
                              </Row>
                              <Row style={{ textAlign: 'start' }}>
                                {multiStepDenseModelHiddenLayers.map(
                                  (value, index) => {
                                    return (
                                      <>
                                        <Col
                                          xs={12}
                                          sm={3}
                                          md='3'
                                          style={{ maxWidth: '210px' }}
                                        >
                                          <label>
                                            Number of neurons Hidden Layer #
                                            {index + 1}
                                          </label>
                                          <Input
                                            type='text'
                                            value={value}
                                            onChange={(e) => {
                                              let temp = []
                                              for (const element of multiStepDenseModelHiddenLayers) {
                                                temp.push(element)
                                              }
                                              temp[index] = parseFloat(
                                                e.target.value
                                              )
                                                ? parseFloat(e.target.value)
                                                : 0
                                              setMultiStepDenseModelHiddenLayers(
                                                temp
                                              )
                                            }}
                                          />
                                        </Col>
                                        <Label
                                          sm={3}
                                          className='label-on-right'
                                          style={{ flex: '0' }}
                                          onClick={() => {
                                            let temp = []
                                            for (const element of multiStepDenseModelHiddenLayers) {
                                              temp.push(element)
                                            }
                                            temp.splice(index, 1)
                                            setMultiStepDenseModelHiddenLayers(
                                              temp
                                            )
                                          }}
                                        >
                                          <div
                                            style={{
                                              marginTop: '57px',
                                              cursor: 'pointer',
                                              color: 'red',
                                              marginLeft: '-23px',
                                            }}
                                          >
                                            <i className='now-ui-icons ui-1_simple-remove' />
                                          </div>
                                        </Label>
                                      </>
                                    )
                                  }
                                )}
                                <Col
                                  xs={12}
                                  sm={3}
                                  md='3'
                                  style={{
                                    maxWidth: '120px',
                                    marginTop: '22px',
                                  }}
                                >
                                  <Button
                                    color='info'
                                    className='btn-round'
                                    onClick={() => {
                                      let temp = []
                                      for (const element of multiStepDenseModelHiddenLayers) {
                                        temp.push(element)
                                      }
                                      temp.push(32)
                                      setMultiStepDenseModelHiddenLayers(temp)
                                    }}
                                    style={{ marginTop: '30px' }}
                                  >
                                    Add{' '}
                                    <i className='now-ui-icons ui-1_simple-add' />
                                  </Button>
                                </Col>
                              </Row>
                            </>
                          )}
                          {selectedModelsArray.includes('recurrentlstm') && (
                            <>
                              {' '}
                              <Row
                                style={{
                                  textAlign: 'center',
                                  marginTop: '40px',
                                }}
                              >
                                <Col xs={12} sm={3} md='12'>
                                  <label
                                    style={{
                                      fontSize: '14px',
                                      fontWeight: 'bolder',
                                      textDecoration: 'underline',
                                    }}
                                  >
                                    LSTM model:
                                  </label>
                                </Col>
                              </Row>
                              <Row
                                style={{
                                  textAlign: 'start',
                                  marginTop: '0px',
                                  paddingBottom: '15px',
                                }}
                              >
                                <Col
                                  xs={12}
                                  sm={3}
                                  md='3'
                                  style={{ maxWidth: '210px' }}
                                >
                                  <label>Window Size</label>
                                  <Input
                                    type='text'
                                    onChange={(e) => {
                                      setLstmModelWindowSize(
                                        parseFloat(e.target.value)
                                          ? parseFloat(e.target.value)
                                          : 0
                                      )
                                    }}
                                    placeholder='12'
                                    value={lstmModelWindowSize}
                                  />
                                </Col>
                              </Row>
                            </>
                          )}
                        </ModalBody>
                        <ModalFooter>
                          <Button
                            color='info'
                            onClick={() => {
                              setDatasetSplitPercentages([
                                parseFloat(datasetSplitPercentages[0]),
                                parseFloat(datasetSplitPercentages[1]),
                                parseFloat(datasetSplitPercentages[2]),
                              ])
                              setIsSettingsModalOpen(false)
                            }}
                          >
                            Save settings
                          </Button>
                        </ModalFooter>
                      </Modal>
                      <Button
                        color='info'
                        className='btn-round'
                        onClick={() => {
                          trainModels((e) => console.log('\n\nCallback: ', e))
                        }}
                        style={{ marginTop: '30px' }}
                      >
                        Begin trainning
                      </Button>
                    </Col>
                  </Row>
                </div>
              </CardBody>
            </Card>
          </Col>
        </Row>
        {trainningTouched && (
          <>
            <Row>
              <Col xs={12} md={6}>
                <Card className='card-chart'>
                  <CardHeader>
                    <h5 className='card-category'>Model metrics MAPE</h5>
                  </CardHeader>
                  <CardBody>
                    <div className='chart-area-lg'>
                      <Bar
                        data={dashboardChartMetrics.data}
                        options={dashboardChartMetrics.options}
                      />
                    </div>
                  </CardBody>
                </Card>
              </Col>
              <Col xs={12} md={6}>
                <Card className='card-chart'>
                  <CardHeader>
                    <h5 className='card-category'>Model metrics MAE</h5>
                  </CardHeader>
                  <CardBody>
                    <div className='chart-area-lg'>
                      <Bar
                        data={dashboardChartMetricsMAE.data}
                        options={dashboardChartMetricsMAE.options}
                      />
                    </div>
                  </CardBody>
                </Card>
              </Col>
            </Row>
            <Row>
              <Col xs={12} md={12}>
                <Card className='card-chart'>
                  <CardHeader>
                    <h5 className='card-category'>
                      Series predictions (TRAINING)
                    </h5>
                    <UncontrolledDropdown>
                      <DropdownToggle
                        className='btn-round btn-icon'
                        color='default'
                        outline
                        style={{ marginTop: '-10px' }}
                        onClick={() => {
                          setPredictionChartFullSize(!predictionChartFullSize)
                        }}
                      >
                        <div style={{ marginTop: '-3px' }}>
                          {predictionChartFullSize === false ? (
                            <BsFullscreen />
                          ) : (
                            <BsFullscreenExit />
                          )}
                        </div>
                      </DropdownToggle>
                    </UncontrolledDropdown>
                  </CardHeader>
                  <CardBody>
                    <div
                      className={
                        predictionChartFullSize === true
                          ? 'chart-area-full'
                          : 'chart-area-lg'
                      }
                    >
                      <Line
                        data={dashboardChartPredictionsTraining.data}
                        options={dashboardChartPredictionsTraining.options}
                      />
                    </div>
                  </CardBody>
                </Card>
              </Col>
              <Col xs={12} md={12}>
                <Card className='card-chart'>
                  <CardHeader>
                    <h5 className='card-category'>
                      Series predictions (VALIDATION)
                    </h5>
                    <UncontrolledDropdown>
                      <DropdownToggle
                        className='btn-round btn-icon'
                        color='default'
                        outline
                        style={{ marginTop: '-10px' }}
                        onClick={() => {
                          setPredictionChartFullSize(!predictionChartFullSize)
                        }}
                      >
                        <div style={{ marginTop: '-3px' }}>
                          {predictionChartFullSize === false ? (
                            <BsFullscreen />
                          ) : (
                            <BsFullscreenExit />
                          )}
                        </div>
                      </DropdownToggle>
                    </UncontrolledDropdown>
                  </CardHeader>
                  <CardBody>
                    <div
                      className={
                        predictionChartFullSize === true
                          ? 'chart-area-full'
                          : 'chart-area-lg'
                      }
                    >
                      <Line
                        data={dashboardChartPredictionsValidation.data}
                        options={dashboardChartPredictionsValidation.options}
                      />
                    </div>
                  </CardBody>
                </Card>
              </Col>
              <Col xs={12} md={12}>
                <Card className='card-chart'>
                  <CardHeader>
                    <h5 className='card-category'>Series predictions (TEST)</h5>
                    <UncontrolledDropdown>
                      <DropdownToggle
                        className='btn-round btn-icon'
                        color='default'
                        outline
                        style={{ marginTop: '-10px' }}
                        onClick={() => {
                          setPredictionChartFullSize(!predictionChartFullSize)
                        }}
                      >
                        <div style={{ marginTop: '-3px' }}>
                          {predictionChartFullSize === false ? (
                            <BsFullscreen />
                          ) : (
                            <BsFullscreenExit />
                          )}
                        </div>
                      </DropdownToggle>
                    </UncontrolledDropdown>
                  </CardHeader>
                  <CardBody>
                    <div
                      className={
                        predictionChartFullSize === true
                          ? 'chart-area-full'
                          : 'chart-area-lg'
                      }
                    >
                      <Line
                        data={dashboardChartPredictions.data}
                        options={dashboardChartPredictions.options}
                      />
                    </div>
                  </CardBody>
                </Card>
              </Col>
            </Row>
          </>
        )}
        {false && (
          <>
            <Row>
              <Col xs={12} md={4}>
                <Card className='card-chart'>
                  <CardHeader>
                    <h5 className='card-category'>Active Users</h5>
                    <CardTitle tag='h2'>34,252</CardTitle>
                    <UncontrolledDropdown>
                      <DropdownToggle
                        className='btn-round btn-icon'
                        color='default'
                        outline
                      >
                        <i className='now-ui-icons loader_gear' />
                      </DropdownToggle>
                      <DropdownMenu right>
                        <DropdownItem>Action</DropdownItem>
                        <DropdownItem>Another Action</DropdownItem>
                        <DropdownItem>Something else here</DropdownItem>
                        <DropdownItem className='text-danger'>
                          Remove data
                        </DropdownItem>
                      </DropdownMenu>
                    </UncontrolledDropdown>
                  </CardHeader>
                  <CardBody>
                    <div className='chart-area'>
                      <Line
                        data={dashboardActiveUsersChart.data}
                        options={dashboardActiveUsersChart.options}
                      />
                    </div>
                    <Table responsive>
                      <tbody>{createTableData()}</tbody>
                    </Table>
                  </CardBody>
                  <CardFooter>
                    <div className='stats'>
                      <i className='now-ui-icons arrows-1_refresh-69' />
                      Just Updated
                    </div>
                  </CardFooter>
                </Card>
              </Col>
              <Col xs={12} md={4}>
                <Card className='card-chart'>
                  <CardHeader>
                    <h5 className='card-category'>Summer Email Campaign</h5>
                    <CardTitle tag='h2'>55,300</CardTitle>
                    <UncontrolledDropdown>
                      <DropdownToggle
                        className='btn-round btn-icon'
                        color='default'
                        outline
                      >
                        <i className='now-ui-icons loader_gear' />
                      </DropdownToggle>
                      <DropdownMenu right>
                        <DropdownItem>Action</DropdownItem>
                        <DropdownItem>Another Action</DropdownItem>
                        <DropdownItem>Something else here</DropdownItem>
                        <DropdownItem className='text-danger'>
                          Remove data
                        </DropdownItem>
                      </DropdownMenu>
                    </UncontrolledDropdown>
                  </CardHeader>
                  <CardBody>
                    <div className='chart-area'>
                      <Line
                        data={dashboardSummerChart.data}
                        options={dashboardSummerChart.options}
                      />
                    </div>
                    <div className='card-progress'>
                      <div className='progress-container'>
                        <span className='progress-badge'>Delivery Rate</span>
                        <Progress max='100' value='90'>
                          <span className='progress-value'>90%</span>
                        </Progress>
                      </div>
                      <div className='progress-container progress-success'>
                        <span className='progress-badge'>Open Rate</span>
                        <Progress max='100' value='60'>
                          <span className='progress-value'>60%</span>
                        </Progress>
                      </div>
                      <div className='progress-container progress-info'>
                        <span className='progress-badge'>Click Rate</span>
                        <Progress max='100' value='12'>
                          <span className='progress-value'>12%</span>
                        </Progress>
                      </div>
                      <div className='progress-container progress-warning'>
                        <span className='progress-badge'>Hard Bounce</span>
                        <Progress max='100' value='5'>
                          <span className='progress-value'>5%</span>
                        </Progress>
                      </div>
                      <div className='progress-container progress-danger'>
                        <span className='progress-badge'>Spam Report</span>
                        <Progress max='100' value='0.11'>
                          <span className='progress-value'>0.11%</span>
                        </Progress>
                      </div>
                    </div>
                  </CardBody>
                  <CardFooter>
                    <div className='stats'>
                      <i className='now-ui-icons arrows-1_refresh-69' />
                      Just Updated
                    </div>
                  </CardFooter>
                </Card>
              </Col>
              <Col xs={12} md={4}>
                <Card className='card-chart'>
                  <CardHeader>
                    <h5 className='card-category'>Active Countries</h5>
                    <CardTitle tag='h2'>105</CardTitle>
                  </CardHeader>
                  <CardBody>
                    <div className='chart-area'>
                      <Line
                        data={dashboardActiveCountriesCard.data}
                        options={dashboardActiveCountriesCard.options}
                      />
                    </div>
                    <VectorMap
                      map={'world_mill'}
                      backgroundColor='transparent'
                      zoomOnScroll={false}
                      containerStyle={{
                        width: '100%',
                        height: '280px',
                      }}
                      containerClassName='map'
                      regionStyle={{
                        initial: {
                          fill: '#e4e4e4',
                          'fill-opacity': 0.9,
                          stroke: 'none',
                          'stroke-width': 0,
                          'stroke-opacity': 0,
                        },
                      }}
                      series={{
                        regions: [
                          {
                            values: mapData,
                            scale: ['#AAAAAA', '#444444'],
                            normalizeFunction: 'polynomial',
                          },
                        ],
                      }}
                    />
                  </CardBody>
                  <CardFooter>
                    <div className='stats'>
                      <i className='now-ui-icons ui-2_time-alarm' />
                      Last 7 days
                    </div>
                  </CardFooter>
                </Card>
              </Col>
            </Row>
            <Row>
              <Col xs={12} md={12}>
                <Card>
                  <CardHeader>
                    <CardTitle tag='h4'>Best Selling Products</CardTitle>
                  </CardHeader>
                  <CardBody>
                    <Table responsive className='table-shopping'>
                      <thead>
                        <tr>
                          <th className='text-center' />
                          <th>PRODUCT</th>
                          <th>COLOR</th>
                          <th>Size</th>
                          <th className='text-right'>PRICE</th>
                          <th className='text-right'>QTY</th>
                          <th className='text-right'>AMOUNT</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr>
                          <td>
                            <div className='img-container'>
                              <img src={jacket} alt='...' />
                            </div>
                          </td>
                          <td className='td-name'>
                            <a href='#jacket'>Suede Biker Jacket</a>
                            <br />
                            <small>by Saint Laurent</small>
                          </td>
                          <td>Black</td>
                          <td>M</td>
                          <td className='td-number'>
                            <small>€</small>3,390
                          </td>
                          <td className='td-number'>1</td>
                          <td className='td-number'>
                            <small>€</small>549
                          </td>
                        </tr>
                        <tr>
                          <td>
                            <div className='img-container'>
                              <img src={shirt} alt='...' />
                            </div>
                          </td>
                          <td className='td-name'>
                            <a href='#shirt'>Jersey T-Shirt</a>
                            <br />
                            <small>by Balmain</small>
                          </td>
                          <td>Black</td>
                          <td>M</td>
                          <td className='td-number'>
                            <small>€</small>499
                          </td>
                          <td className='td-number'>2</td>
                          <td className='td-number'>
                            <small>€</small>998
                          </td>
                        </tr>
                        <tr>
                          <td>
                            <div className='img-container'>
                              <img src={swim} alt='...' />
                            </div>
                          </td>
                          <td className='td-name'>
                            <a href='#pants'>Slim-Fit Swim Short </a>
                            <br />
                            <small>by Prada</small>
                          </td>
                          <td>Red</td>
                          <td>M</td>
                          <td className='td-number'>
                            <small>€</small>200
                          </td>
                          <td className='td-number'>3</td>
                          <td className='td-number'>
                            <small>€</small>799
                          </td>
                        </tr>
                        <tr>
                          <td colSpan='5' />
                          <td className='td-total'>Total</td>
                          <td className='td-price'>
                            <small>€</small>2,346
                          </td>
                        </tr>
                      </tbody>
                    </Table>
                  </CardBody>
                </Card>
              </Col>
            </Row>
          </>
        )}
      </div>
    </>
  )
}

export default Dashboard
