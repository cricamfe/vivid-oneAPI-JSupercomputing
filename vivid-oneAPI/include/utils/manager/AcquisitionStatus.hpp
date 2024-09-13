#ifndef ACQUISITION_STATUS_HPP
#define ACQUISITION_STATUS_HPP

enum class AcquisitionStatus {
    AcquiredCore, // Se adquirió un core directamente
    Enqueued,     // La tarea fue encolada
    Failed        // La adquisición falló (sin cores y sin espacio en la cola)
};

#endif // ACQUISITION_STATUS_HPP