import React, { useState } from 'react';
import { useAuth } from './contexts/AuthContext';

// Layout & Pages
import AppLayout from './components/layout/AppLayout';
import LoginPage from './pages/Auth/LoginPage';
import DashboardPage from './pages/Dashboard/DashboardPage';
import ClinicScreeningPage from './pages/Clinic/ClinicScreeningPage';
import PatientsPage from './pages/Patients/PatientsPage';
import RecallsPage from './pages/Recalls/RecallsPage';
import PatientPortal from './components/PatientPortal';
import ProjectEvidencePage from './components/ProjectEvidencePage';

export default function App() {
  const { currentUser, isAuthLoading, logout } = useAuth();
  const [activeTab, setActiveTab] = useState('dashboard');
  const [screeningPatient, setScreeningPatient] = useState(null);

  // 1. Loading State
  if (isAuthLoading) {
    return (
      <div style={{ display: 'flex', height: '100vh', width: '100vw', alignItems: 'center', justifyContent: 'center', backgroundColor: 'var(--background)' }}>
        <div style={{ width: '40px', height: '40px', border: '4px solid rgba(13, 148, 136, 0.1)', borderTopColor: 'var(--primary)', borderRadius: '50%', animation: 'spin 1s linear infinite' }} />
        <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
      </div>
    );
  }

  // 2. Unauthenticated State
  if (!currentUser) {
    return <LoginPage />;
  }

  // 3. Patient Portal View
  if (currentUser.role === 'patient') {
    return <PatientPortal currentUser={currentUser} onLogout={logout} />;
  }

  // 4. Main Medical Staff Workspace
  return (
    <AppLayout activeTab={activeTab} onTabChange={setActiveTab}>
      {activeTab === 'dashboard' && <DashboardPage />}

      {activeTab === 'screening' && (
        <ClinicScreeningPage
          initialPatient={screeningPatient}
          onClearPatient={() => setScreeningPatient(null)}
        />
      )}

      {activeTab === 'patients' && (
        <PatientsPage
          onStartScreening={(patient) => {
            setScreeningPatient(patient);
            setActiveTab('screening');
          }}
        />
      )}

      {activeTab === 'recalls' && <RecallsPage />}

      {activeTab === 'evidence' && <ProjectEvidencePage />}
    </AppLayout>
  );
}
