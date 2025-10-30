pipeline{
    agent any

    environment {
        VENV_DIR = 'venv'
        GCP_PROJECT= "dliva2-audc8s"
        GCLOUD_PATH= "/var/jenkins_home/google-cloud-sdk/bin"
    }

    stages{
        stage('Cloning Github repo to Jenkins'){
            steps{
                script{
                    echo 'Cloning Github repo to Jenkins................'
                    checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'github-token', url: 'https://github.com/manish182003/MLOPS-COURSE-PROJECT-1.git']])
                    
                }
            }
        }

         stage('Setting Up or virtual enviroment and installing dependencies'){
            steps{
                script{
                    echo 'Setting Up or virtual enviroment and installing dependencies................'
                    sh '''
                    python -m venv ${VENV_DIR}
                    . ${VENV_DIR}/bin/activate
                    pip install --upgrade pip
                    pip install -e .
                    '''
                   
                    
                }
            }
        }

        

         stage('Building and Pushing Docker Image to GCR') {
      steps {
        withCredentials([string(credentialsId: 'gcp-user-auth', variable: 'GCP_JSON')]) {
          echo  'Building and Pushing Docker Image to GCR'
          sh '''
            export PATH=$PATH:${GCLOUD_PATH}
            gcloud config set project ${GCP_PROJECT}
            gcloud auth configure-docker --quiet


          docker build -t gcr.io/${GCP_PROJECT}/ml-project-latest .

          docker push  gcr.io/${GCP_PROJECT}/ml-project-latest 

          '''
        }
      }
    }


   

    }
}