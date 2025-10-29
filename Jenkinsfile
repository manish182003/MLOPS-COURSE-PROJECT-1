pipeline{
    agent any

    enviroment {
        VENV_DIR = 'venv'
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
    }
}