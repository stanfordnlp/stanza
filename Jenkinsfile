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
      steps {
        sh 'ls -lhd /var/jenkins_home/test_resources/*'
      }
    }

  }
}