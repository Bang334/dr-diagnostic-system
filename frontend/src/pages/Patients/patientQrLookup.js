export async function resolvePatientFromQr(decodedText, fetchPatients) {
  const patientCode = typeof decodedText === 'string' ? decodedText.trim() : '';
  if (!patientCode) return null;

  const patients = await fetchPatients(patientCode);
  if (!Array.isArray(patients)) return null;

  const normalizedCode = patientCode.toLocaleLowerCase('vi-VN');
  return patients.find((patient) => (
    patient?.patient_code?.trim().toLocaleLowerCase('vi-VN') === normalizedCode
  )) || null;
}
