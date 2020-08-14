pipeline {
  agent any
  stages {
    stage('Build') {
      steps {
        sh '''echo "hello world"
'''
      }
    }

    stage('Test') {
      environment {
        CI = 'true'
      }
      steps {
        sh 'echo "Done testing!"'
      }
    }

  }
}